#!/usr/bin/env python3
"""M3: Evaluate base model with search/read tools on eval questions.

Measures tool-calling ability and answer accuracy BEFORE any RL training.
This is the baseline we compare against after training.

Usage:
    python scripts/run_with_tools.py --eval-data data/eval.jsonl
    python scripts/run_with_tools.py --eval-data data/eval.jsonl --model Qwen/Qwen3-0.6B --max-rounds 10
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.env.search_env import SearchEnvironment
from src.utils.device import get_device, get_dtype

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web and return results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a web page and find sections matching keywords. "
                "Returns up to 5 excerpts containing the keywords with surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to read"},
                    "keywords": {"type": "string", "description": "Keywords to search for within the page"},
                },
                "required": ["url", "keywords"],
            },
        },
    },
]


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output."""
    calls = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "name" in call and "arguments" in call:
                calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


def extract_answer(text: str) -> str | None:
    """Extract answer from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def normalize(s: str) -> str:
    """Normalize for comparison: lowercase, strip articles/punct."""
    s = s.lower().strip()
    for article in ["the ", "a ", "an "]:
        if s.startswith(article):
            s = s[len(article):]
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def check_answer(predicted: str | None, ground_truth: str, aliases: list[str] | None = None) -> bool:
    """Check if predicted answer matches ground truth (normalized)."""
    if predicted is None:
        return False
    pred_norm = normalize(predicted)
    targets = [ground_truth] + (aliases or [])
    for target in targets:
        t_norm = normalize(target)
        if pred_norm == t_norm or t_norm in pred_norm or pred_norm in t_norm:
            return True
    return False


def dispatch_tool(env: SearchEnvironment, name: str, args: dict) -> str:
    if name == "search":
        return env.search(query=args["query"], max_results=args.get("max_results", 5))
    elif name == "read":
        return env.read(url=args["url"], keywords=args["keywords"])
    return f"[Unknown tool: {name}]"


def run_single_question(
    model, tokenizer, env: SearchEnvironment,
    question: str, device: str, dtype: torch.dtype,
    max_rounds: int = 10, max_new_tokens: int = 512,
) -> dict:
    """Run model on a single question with tool access. Returns result dict."""
    env.reset()

    system_msg = (
        "You are a research assistant. Use the search and read tools "
        "to find information. When you have the answer, provide it "
        "inside <answer></answer> tags."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]

    tool_calls_made = []
    final_answer = None

    for round_num in range(1, max_rounds + 1):
        # Build prompt
        text = tokenizer.apply_chat_template(
            messages, tools=TOOLS_SPEC, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"    round {round_num}: {response_text[:120].replace(chr(10), ' ')}", flush=True)

        # Check for tool calls
        calls = parse_tool_calls(response_text)

        if calls:
            # Add assistant message with tool calls
            messages.append({"role": "assistant", "content": response_text})

            for call in calls:
                tool_name = call["name"]
                tool_args = call["arguments"]
                tool_calls_made.append({"name": tool_name, "arguments": tool_args})

                print(f"      → {tool_name}({json.dumps(tool_args)[:80]})", flush=True)
                result = dispatch_tool(env, tool_name, tool_args)
                print(f"      ← {result[:80].replace(chr(10), ' ')}...", flush=True)

                messages.append({"role": "tool", "content": result})
        else:
            # No tool calls — check for answer
            final_answer = extract_answer(response_text)
            if final_answer is None and response_text.strip():
                # Model gave text but no <answer> tags — use raw text as answer
                final_answer = response_text.strip()
            break

    return {
        "answer": final_answer,
        "tool_calls": tool_calls_made,
        "num_rounds": round_num if 'round_num' in dir() else 0,
        "used_tools": len(tool_calls_made) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="M3: Evaluate model with tools")
    parser.add_argument("--eval-data", type=str, default="data/eval.jsonl", help="Eval JSONL path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max tool-calling rounds per question")
    parser.add_argument("--output", type=str, default=None, help="Output results JSON")
    args = parser.parse_args()

    device = get_device()
    dtype = get_dtype()
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {args.model}")
    print()

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    print("Model loaded.\n", flush=True)

    # Load eval data
    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        print(f"Error: {eval_path} not found")
        return

    examples = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} eval questions\n")

    env = SearchEnvironment()
    results = []

    for i, ex in enumerate(examples):
        question = ex["prompt"][1]["content"]  # user message
        ground_truth = ex["answer"]
        num_hops = ex.get("num_hops", "?")

        print(f"[{i+1}/{len(examples)}] ({num_hops}-hop) {question}", flush=True)
        print(f"  Expected: {ground_truth}", flush=True)

        start = time.time()
        result = run_single_question(
            model, tokenizer, env, question, device, dtype,
            max_rounds=args.max_rounds,
        )
        elapsed = time.time() - start

        correct = check_answer(result["answer"], ground_truth, ex.get("answer_aliases"))

        print(f"  Got: {result['answer']}")
        print(f"  Correct: {correct} | Tools used: {len(result['tool_calls'])} | "
              f"Rounds: {result['num_rounds']} | Time: {elapsed:.1f}s")
        print()

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": result["answer"],
            "correct": correct,
            "num_hops": num_hops,
            "tool_calls": result["tool_calls"],
            "num_rounds": result["num_rounds"],
            "used_tools": result["used_tools"],
            "time": round(elapsed, 1),
        })

    # Summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    used_tools_count = sum(1 for r in results if r["used_tools"])

    print(f"Overall accuracy: {correct_count}/{total} ({100*correct_count/total:.0f}%)")
    print(f"Used tools: {used_tools_count}/{total} ({100*used_tools_count/total:.0f}%)")

    # By hop count
    hop_groups = {}
    for r in results:
        h = r["num_hops"]
        if h not in hop_groups:
            hop_groups[h] = []
        hop_groups[h].append(r)

    print("\nBy hop count:")
    for h in sorted(hop_groups.keys()):
        group = hop_groups[h]
        gc = sum(1 for r in group if r["correct"])
        gt = sum(1 for r in group if r["used_tools"])
        print(f"  {h}-hop: {gc}/{len(group)} correct, {gt}/{len(group)} used tools")

    # Save results
    output_path = Path(args.output) if args.output else Path("results/m3_baseline_with_tools.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "total": total,
            "correct": correct_count,
            "accuracy": round(correct_count / total, 3) if total else 0,
            "used_tools": used_tools_count,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
