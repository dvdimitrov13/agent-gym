#!/usr/bin/env python3
"""Run eval completions using a trained checkpoint.

Loads the base model + LoRA adapter, runs each eval question through
the tool-calling loop, and saves full trajectories as JSON.

Usage:
    python scripts/run_eval_completions.py \
        --checkpoint checkpoints/qwen3-14b-grpo-v2/checkpoint-600 \
        --eval-data data/eval_trl_v2.jsonl \
        --output results/eval_600.json
"""

import argparse
import json
import logging
import re
import time
import sys
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.tito import (
    _init_token_ids, _find_tool_call, _parse_tool_call_json,
    _encode_tool_result, strip_thinking_tokens, _TOOL_CALL_END_ID,
)
from src.env.search_env import SearchEnvironment


class EvalToolForceProcessor:
    """LogitsProcessor for eval: hard-forces tool calls after thinking.

    1. Suppress ALL EOS tokens inside <think> blocks (Qwen3 has two: 151645, 151643)
    2. Hard cap thinking at max_thinking_tokens (forces </think>)
    3. After </think>, hard-force newline then <tool_call>
    """

    def __init__(self, tokenizer, max_thinking_tokens=384):
        self.max_thinking_tokens = max_thinking_tokens
        self.think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        self.think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.tool_call_id = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        # Qwen3 has TWO eos tokens (151645 + 151643) — suppress both inside think
        # tokenizer.eos_token_id only returns one; generation_config has both
        self.eos_ids = [151645, 151643]  # Qwen3 specific
        self._in_think = {}
        self._think_tokens = {}
        self._state = {}  # None | "need_newline" | "need_tool_call"

    def reset(self, assume_in_think=True):
        self._in_think = {}
        self._think_tokens = {}
        self._state = {}
        self._assume_in_think = assume_in_think

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            last_token = input_ids[i, -1].item()

            # After </think>, hard-force \n then soft boost <tool_call>
            state = self._state.get(i)
            if state == "need_newline":
                scores[i, :] = float('-inf')
                scores[i, self.newline_id] = 0
                self._state[i] = "need_tool_call"
                continue
            elif state == "need_tool_call":
                scores[i, self.tool_call_id] += 15.0
                self._state[i] = None
                continue

            # Track think blocks
            if last_token == self.think_start_id:
                self._in_think[i] = True
                self._think_tokens[i] = 0
            elif i not in self._in_think and getattr(self, '_assume_in_think', True):
                self._in_think[i] = True
                self._think_tokens[i] = 0

            if last_token == self.think_end_id:
                self._in_think[i] = False
                self._state[i] = "need_newline"
                continue

            # Inside think block
            if self._in_think.get(i, False):
                self._think_tokens[i] = self._think_tokens.get(i, 0) + 1

                # Suppress ALL EOS tokens inside think blocks
                for eos_id in self.eos_ids:
                    scores[i, eos_id] = float('-inf')

                # Soft nudge only (matches training — no hard cap)
                if self._think_tokens[i] >= self.max_thinking_tokens // 2:
                    remaining = self.max_thinking_tokens - self.max_thinking_tokens // 2
                    progress = (self._think_tokens[i] - self.max_thinking_tokens // 2) / max(remaining, 1)
                    boost = min(2.0 + 13.0 * progress, 15.0)
                    scores[i, self.think_end_id] += boost

        return scores


def load_model(checkpoint_path, base_model="Qwen/Qwen3-14B"):
    """Load base model, optionally with LoRA adapter."""
    if checkpoint_path and checkpoint_path != "NONE":
        logger.info(f"Loading tokenizer from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        logger.info(f"Loading tokenizer from {base_model} (no adapter)...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

    if checkpoint_path and checkpoint_path != "NONE":
        logger.info(f"Loading LoRA adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        logger.info("Model loaded and merged.")
    else:
        logger.info("Base model loaded (no LoRA adapter).")
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, question_text):
    """Build prompt token IDs from question with few-shot examples."""
    system = (
        "You are a research assistant. Use the tools below to find information and submit relevant passages.\n\n"
        "Tools:\n\n"
        "1. search(query: str, max_results: int = 5) -> str\n"
        "   Search the web for a query. Returns a list of snippets, each tagged with an ID like [S1], [S2], etc.\n"
        "   Use specific, targeted queries. For multi-hop questions, search for one fact at a time.\n\n"
        "2. read(url: str, keywords: str) -> str\n"
        "   Fetch a webpage and extract sections matching the keywords. Returns excerpts tagged with IDs like [R1], [R2].\n"
        "   Use this to get more detail from a promising search result.\n\n"
        "3. submit_answer(passage_ids: list[str]) -> str\n"
        "   Submit your final answer as a ranked list of passage IDs (most relevant first).\n"
        "   IDs must be snippet IDs from search/read results (e.g. [\"S1\", \"S3\", \"R2\"]).\n\n"
        "Always end your turn with a tool call. When you have enough information, call submit_answer."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question_text},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_ids


def parse_raw_text_tool_call(text):
    """Parse tool calls in raw text format: search("query") or submit_answer([...])."""
    patterns = [
        (r'search\(\s*(?:query\s*=\s*)?["\'](.+?)["\'](?:\s*,\s*max_results\s*=\s*(\d+))?\s*\)', "search"),
        (r'read\(\s*(?:url\s*=\s*)?["\'](.+?)["\'](?:\s*,\s*(?:keywords\s*=\s*)?["\'](.+?)["\'])?\s*\)', "read"),
        (r'submit_answer\(\s*(\[.+?\])\s*\)', "submit_answer"),
    ]
    for pattern, tool_name in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            if tool_name == "search":
                return tool_name, {"query": m.group(1), "max_results": int(m.group(2) or 5)}
            elif tool_name == "read":
                return tool_name, {"url": m.group(1), "keywords": m.group(2) or ""}
            elif tool_name == "submit_answer":
                try:
                    ids = json.loads(m.group(1).replace("'", '"'))
                except:
                    ids = [s.strip().strip("'\"") for s in m.group(1).strip("[]").split(",")]
                # Ensure all IDs are strings
                ids = [str(x) for x in ids]
                return tool_name, {"passage_ids": ids}
    return None, None


def run_single_eval(model, tokenizer, question_text, thinking_processor=None,
                    max_iterations=10, max_new_tokens=1024, disable_thinking=False):
    """Run a single question through the tool-calling loop. Returns trajectory."""
    _eval_start_time = time.time()
    _init_token_ids(tokenizer)
    env = SearchEnvironment(use_cache=False)
    device = next(model.parameters()).device

    prompt_ids = build_prompt(tokenizer, question_text)
    if disable_thinking:
        think_prefix = tokenizer.encode("<think>\n</think>", add_special_tokens=False)
    else:
        think_prefix = tokenizer.encode("<think>\n", add_special_tokens=False)

    trajectory = []
    completion_ids = []
    snippet_store = {}
    total_gen_time = 0.0
    total_tool_time = 0.0

    for iteration in range(max_iterations):
        # Build context: prompt + stripped completion + think prefix
        if completion_ids:
            ctx = strip_thinking_tokens(completion_ids)
        else:
            ctx = []
        full_input = prompt_ids + ctx + think_prefix

        input_ids = torch.tensor([full_input], device=device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if thinking_processor is not None:
            thinking_processor.reset(assume_in_think=not disable_thinking)
            gen_kwargs["logits_processor"] = [thinking_processor]

        t_gen = time.time()
        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)
        gen_elapsed = time.time() - t_gen
        total_gen_time += gen_elapsed

        new_tokens = outputs[0, len(full_input):].tolist()
        new_tokens = list(think_prefix) + new_tokens
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        logger.info(f"  Iter {iteration} gen ({len(new_tokens)} tokens)")
        logger.info(f"    START: {new_text[:200]}")
        logger.info(f"    END:   {new_text[-300:]}")

        # Try XML format first (training format)
        name, args = None, None
        span = _find_tool_call(new_tokens)
        if span:
            parsed = _parse_tool_call_json(tokenizer, new_tokens, span[0], span[1])
            if parsed:
                name, args = parsed
                logger.info(f"  Parsed XML tool call: {name}")

        # Fallback: raw text format (model's natural output after </think>)
        if name is None:
            name, args = parse_raw_text_tool_call(new_text)
            if name:
                logger.info(f"  Parsed raw-text tool call: {name}")

        if name is None:
            trajectory.append({
                "iteration": iteration,
                "type": "no_tool_call",
                "text": new_text,
            })
            logger.info(f"  Iter {iteration}: no tool call found, stopping")
            break

        # Execute tool — normalize args before dispatch
        t_tool = time.time()
        try:
            if name == "search":
                result = env.search(**args)
            elif name == "read":
                # Ensure keywords is a string
                if "keywords" in args and isinstance(args["keywords"], list):
                    args["keywords"] = " ".join(str(k) for k in args["keywords"])
                result = env.read(**args)
            elif name == "submit_answer":
                result = env.submit_answer(**args)
            else:
                result = f"Unknown tool: {name}"
        except (TypeError, Exception) as e:
            logger.warning(f"  Tool call error: {e}")
            result = f"Error: {e}"
        tool_elapsed = time.time() - t_tool
        total_tool_time += tool_elapsed
        logger.info(f"  Timing: gen={gen_elapsed:.1f}s tool={tool_elapsed:.1f}s ({name})")

        # Parse snippets from result
        from src.training.tito_trainer import _parse_result_snippets
        _parse_result_snippets(result, snippet_store)

        trajectory.append({
            "iteration": iteration,
            "type": "tool_call",
            "tool": name,
            "args": args,
            "result": result[:500],
            "generation_text": new_text[:500],
        })

        args_short = json.dumps(args)[:80]
        logger.info(f"  Iter {iteration}: {name}({args_short})")

        if name == "submit_answer":
            logger.info(f"  -> Submitted: {args}")
            break

        # Splice tool result into completion_ids for next iteration
        # For XML tool calls, find </tool_call> end; for raw text, use full tokens
        tc_end = len(new_tokens)
        for j in range(len(new_tokens) - 1, -1, -1):
            if new_tokens[j] == _TOOL_CALL_END_ID:
                tc_end = j + 1
                break

        kept = new_tokens[:tc_end]
        splice = _encode_tool_result(tokenizer, result)
        completion_ids = completion_ids + kept + splice

    elapsed = time.time() - _eval_start_time
    logger.info(f"  Total: {elapsed:.1f}s (gen={total_gen_time:.1f}s tool={total_tool_time:.1f}s)")
    return {
        "trajectory": trajectory,
        "snippet_store": snippet_store,
        "num_iterations": len(trajectory),
        "submitted": any(t.get("tool") == "submit_answer" for t in trajectory),
        "submitted_ids": next((t["args"] for t in trajectory if t.get("tool") == "submit_answer"), None),
        "latency_seconds": round(elapsed, 1),
        "gen_time_seconds": round(total_gen_time, 1),
        "tool_time_seconds": round(total_tool_time, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Only run first N examples (0=all)")
    parser.add_argument("--disable-thinking", action="store_true", help="Prepend empty think block instead of open think")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.base_model)

    # Eval processor: suppress dual EOS, hard-force \n after </think>, +15 <tool_call>
    if args.disable_thinking:
        thinking_processor = EvalToolForceProcessor(tokenizer, max_thinking_tokens=1)
        logger.info(f"EvalToolForceProcessor: thinking DISABLED, eos_ids={thinking_processor.eos_ids}")
    else:
        thinking_processor = EvalToolForceProcessor(tokenizer, max_thinking_tokens=384)
        logger.info(f"EvalToolForceProcessor: max=384, eos_ids={thinking_processor.eos_ids}")

    eval_examples = []
    with open(args.eval_data) as f:
        for line in f:
            if line.strip():
                eval_examples.append(json.loads(line))
    if args.limit > 0:
        eval_examples = eval_examples[:args.limit]
    logger.info(f"Running {len(eval_examples)} eval examples")

    results = []
    t0 = time.time()

    for i, ex in enumerate(eval_examples):
        question = None
        for msg in ex["prompt"]:
            if msg["role"] == "user":
                question = msg["content"]
                break

        logger.info(f"\n[{i+1}/{len(eval_examples)}] {question[:80]}...")

        result = run_single_eval(
            model, tokenizer, question,
            thinking_processor=thinking_processor,
            max_iterations=args.max_iterations,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=args.disable_thinking,
        )
        result["question"] = question
        result["gold_answer"] = ex.get("answer")
        result["gold_passages"] = ex.get("gold_passages")
        result["sub_answers"] = ex.get("sub_answers")
        result["num_hops"] = ex.get("num_hops")
        results.append(result)

        status = "SUBMITTED" if result["submitted"] else "NO SUBMIT"
        logger.info(f"  -> {status} in {result['num_iterations']} iterations")

    elapsed = time.time() - t0
    logger.info(f"\nEval complete: {len(results)} examples in {elapsed:.0f}s")

    submitted = sum(1 for r in results if r["submitted"])
    logger.info(f"Submit rate: {submitted}/{len(results)} ({100*submitted/len(results):.0f}%)")

    # Score with LLM judge
    logger.info("Running LLM judge on all results...")
    from src.rewards.llm_judge_reward import _get_client, _judge_single, JUDGE_PROMPT
    client = _get_client()
    if client:
        for i, result in enumerate(results):
            if not result["submitted"]:
                result["judge_score"] = 0.0
                continue
            # Build passages text from snippet store
            ids = result.get("submitted_ids", {})
            if isinstance(ids, dict):
                ids = ids.get("passage_ids", [])
            passages_lines = []
            for pid in (ids or []):
                content = result.get("snippet_store", {}).get(pid, "(content not found)")
                passages_lines.append(f"[{pid}] {content[:200]}")
            passages_text = "\n".join(passages_lines) if passages_lines else "(no passages)"

            # Build trajectory text
            traj_lines = []
            for t in result["trajectory"]:
                tool = t.get("tool")
                if tool == "search":
                    traj_lines.append(f'-> search("{t["args"].get("query", "?")}")')
                elif tool == "read":
                    traj_lines.append(f'-> read({t["args"].get("url", "?")[:60]})')
                elif tool == "submit_answer":
                    traj_lines.append(f'-> submit_answer({ids})')
            trajectory_text = "\n".join(traj_lines) or "(no actions)"

            score = _judge_single(client, result["question"], trajectory_text, passages_text)
            result["judge_score"] = score
            logger.info(f"  [{i+1}] judge={score:.3f} Q: {result['question'][:60]}")

        scores = [r.get("judge_score", 0) for r in results]
        logger.info(f"Judge avg: {sum(scores)/len(scores):.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
