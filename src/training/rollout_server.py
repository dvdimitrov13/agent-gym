#!/usr/bin/env python3
"""Off-policy rollout server for TI/TO GRPO training.

Runs on GPU 0. Continuously generates TI/TO rollouts in background threads,
serves completed rollouts to the trainer on GPU 1 via HTTP.

Continuously generates TI/TO rollouts in background threads and serves
completed rollouts to the trainer via HTTP. The trainer pulls when ready.

Architecture (OLMo-3 / PipelineRL style):
  Trainer (GPU 1) → /submit_prompts → prompt_queue
  Actor worker     → prompt_queue → generate TI/TO → result_queue
  Trainer (GPU 1) → /fetch_rollouts ← result_queue
  Trainer (GPU 1) → /update_weights → hot-reload LoRA

Endpoints:
  POST /submit_prompts   — push prompts for generation
  POST /fetch_rollouts   — pull completed rollouts (blocks until ready)
  POST /update_weights   — reload LoRA adapter
  GET  /health           — status + queue sizes
  GET  /v1/models        — TRL compatibility

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m src.training.rollout_server \
      --model Qwen/Qwen3-0.6B --port 8000
"""

import argparse
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI()

# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_base_model = None
_device = None
_lock = threading.Lock()  # protects model access during weight updates

# Queues
_prompt_queue: deque = deque()
_result_queue: deque = deque()
_prompt_event = threading.Event()  # signals new prompts available
_result_event = threading.Event()  # signals new results available

# Config (set at startup)
_gen_config = {
    "max_new_tokens": 512,
    "max_tool_iterations": 6,
    "temperature": 1.0,
    "top_p": 0.9,
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RolloutRequest:
    """A prompt submitted for rollout generation."""
    prompt_ids: list[int]
    prompt_text: str  # for logging
    metadata: dict = field(default_factory=dict)  # forwarded to trainer (answer, etc.)
    batch_id: int = 0


@dataclass
class RolloutResult:
    """A completed rollout ready for training."""
    prompt_ids: list[int]
    completion_ids: list[int]
    tool_mask: list[int]  # 1=model, 0=tool-result
    completions: list[dict]  # message format for reward functions
    snippet_store: dict  # snippet_id → content
    tool_call_count: int
    behavior_logprobs: list[float] | None  # log P(token) under behavior policy
    metadata: dict  # forwarded from request
    batch_id: int
    generation_time: float


# ── Pydantic models for HTTP ─────────────────────────────────────────────────

class SubmitPromptsRequest(BaseModel):
    prompts: list[dict]  # [{prompt_ids, prompt_text, metadata, batch_id}]
    gen_config: dict | None = None  # override generation config


class FetchRolloutsRequest(BaseModel):
    count: int = 4  # how many rollouts to fetch
    timeout: float = 120.0  # max seconds to wait


class FetchRolloutsResponse(BaseModel):
    rollouts: list[dict]
    queue_size: int


class UpdateWeightsRequest(BaseModel):
    lora_path: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "prompt_queue": len(_prompt_queue),
        "result_queue": len(_result_queue),
    }


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "rollout-server", "object": "model"}]}


@app.post("/submit_prompts")
def submit_prompts(req: SubmitPromptsRequest):
    global _gen_config
    if req.gen_config:
        _gen_config.update(req.gen_config)

    for p in req.prompts:
        _prompt_queue.append(RolloutRequest(
            prompt_ids=p["prompt_ids"],
            prompt_text=p.get("prompt_text", ""),
            metadata=p.get("metadata", {}),
            batch_id=p.get("batch_id", 0),
        ))
    _prompt_event.set()

    logger.info(f"Submitted {len(req.prompts)} prompts (queue: {len(_prompt_queue)})")
    return {"status": "ok", "queue_size": len(_prompt_queue)}


@app.post("/fetch_rollouts")
def fetch_rollouts(req: FetchRolloutsRequest):
    """Fetch completed rollouts. Blocks until enough are ready or timeout."""
    deadline = time.time() + req.timeout

    while len(_result_queue) < req.count and time.time() < deadline:
        _result_event.wait(timeout=1.0)
        _result_event.clear()

    results = []
    while results.__len__() < req.count and _result_queue:
        r = _result_queue.popleft()
        results.append({
            "prompt_ids": r.prompt_ids,
            "completion_ids": r.completion_ids,
            "tool_mask": r.tool_mask,
            "completions": r.completions,
            "snippet_store": r.snippet_store,
            "tool_call_count": r.tool_call_count,
            "behavior_logprobs": r.behavior_logprobs,
            "metadata": r.metadata,
            "batch_id": r.batch_id,
            "generation_time": r.generation_time,
        })

    return FetchRolloutsResponse(rollouts=results, queue_size=len(_result_queue))


@app.post("/update_weights")
def update_weights(req: UpdateWeightsRequest):
    global _model, _base_model
    from peft import PeftModel

    if not os.path.exists(req.lora_path):
        return {"status": "error", "message": f"Path not found: {req.lora_path}"}

    t0 = time.time()
    try:
        with _lock:
            del _model
            torch.cuda.empty_cache()
            _model = PeftModel.from_pretrained(_base_model, req.lora_path)
            _model.eval()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        logger.info(f"Weights updated from {req.lora_path} in {elapsed:.1f}s")
        return {"status": "ok", "time": elapsed}
    except Exception as e:
        logger.error(f"Weight update failed: {e}")
        _model = _base_model
        return {"status": "error", "message": str(e)}


# ── Actor worker (background thread) ────────────────────────────────────────

def _actor_worker():
    """Background thread: pulls prompts from queue, generates TI/TO rollouts."""
    from src.training.tito import (
        _init_token_ids, _find_tool_call, _parse_tool_call_json,
        _encode_tool_result, strip_thinking_tokens,
        _TOOL_CALL_END_ID,
    )
    from src.env.search_env import SearchEnvironment

    def _parse_result_snippets(result_text, store):
        """Parse snippet IDs ([S1], [R2], etc.) and content from tool result."""
        import re as _re
        current_id = None
        current_text = []
        for line in result_text.split("\n"):
            m = _re.match(r'^\[([SR]\d+)\]', line)
            if m:
                if current_id and current_text:
                    store[current_id] = " ".join(current_text)
                current_id = m.group(1)
                rest = line[m.end():].strip()
                current_text = [rest] if rest else []
            elif current_id and line.strip():
                current_text.append(line.strip())
            elif not line.strip():
                if current_id and current_text:
                    store[current_id] = " ".join(current_text)
                current_id = None
                current_text = []
        if current_id and current_text:
            store[current_id] = " ".join(current_text)

    _init_token_ids(_tokenizer)
    env = SearchEnvironment(use_cache=False)

    logger.info("Actor worker started")

    while True:
        # Wait for prompts
        if not _prompt_queue:
            _prompt_event.wait(timeout=5.0)
            _prompt_event.clear()
            continue

        # Grab a prompt
        try:
            request = _prompt_queue.popleft()
        except IndexError:
            continue

        try:
            _process_rollout(request, env, _parse_result_snippets, config_ref=_gen_config)
        except Exception as e:
            logger.error(f"Actor error: {e}", exc_info=True)
            continue


def _process_rollout(request, env, _parse_result_snippets, config_ref):
    """Process a single rollout request. Separated for error isolation."""
    from src.training.tito import (
        _find_tool_call, _parse_tool_call_json,
        _encode_tool_result, strip_thinking_tokens,
        _TOOL_CALL_END_ID,
    )

    global _model, _tokenizer, _device, _lock, _result_queue, _result_event

    t0 = time.time()
    env.reset()

    prompt_ids = request.prompt_ids
    prompt_ids = [int(x) for x in prompt_ids]  # ensure ints from JSON
    completion_ids = []
    tool_mask = []
    completions_msgs = []
    snippet_store = {}
    tool_call_count = 0

    config = config_ref.copy()
    pad_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id

    # Think prefix for disable_thinking mode
    think_prefix = _tokenizer.encode("<think>\n</think>", add_special_tokens=False)

    # Initial generation
    current_prompt = prompt_ids + think_prefix
    input_ids = torch.tensor([current_prompt], device=_device)
    attention_mask = torch.ones_like(input_ids)

    with _lock:
        with torch.no_grad():
            outputs = _model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                do_sample=True,
                pad_token_id=pad_id,
            )

    new_tokens = outputs[0, len(current_prompt):].tolist()
    while new_tokens and new_tokens[-1] == pad_id:
        new_tokens.pop()

    completion_ids = list(think_prefix) + new_tokens
    tool_mask = [1] * len(completion_ids)
    new_tokens_start = 0

    # TI/TO loop
    for iteration in range(config["max_tool_iterations"]):
        search_portion = completion_ids[new_tokens_start:]
        span = _find_tool_call(search_portion)
        if span:
            span = (span[0] + new_tokens_start, span[1] + new_tokens_start)

        if not span:
            break

        parsed = _parse_tool_call_json(_tokenizer, completion_ids, span[0], span[1])
        if not parsed:
            break

        name, args = parsed
        tool_call_count += 1

        try:
            if name == "search":
                result = env.search(**args)
            elif name == "read":
                kw = args.get("keywords", "")
                if isinstance(kw, list):
                    args["keywords"] = " ".join(str(k) for k in kw)
                result = env.read(**args)
            elif name == "submit_answer":
                ids = args.get("passage_ids", args if isinstance(args, list) else [])
                ids = [str(x) for x in ids]
                result = env.submit_answer(passage_ids=ids)
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Error: {e}"

        _parse_result_snippets(result, snippet_store)

        tc_end = len(completion_ids)
        for j in range(len(completion_ids) - 1, -1, -1):
            if completion_ids[j] == _TOOL_CALL_END_ID:
                tc_end = j + 1
                break

        kept = completion_ids[:tc_end]
        splice = _encode_tool_result(_tokenizer, result)
        tool_mask = tool_mask[:tc_end] + [0] * len(splice)
        completion_ids = kept + splice

        completions_msgs.append({"role": "assistant", "content": _tokenizer.decode(kept, skip_special_tokens=True)})
        completions_msgs.append({"role": "tool", "content": result})

        if name == "submit_answer":
            break

        # Generate next turn
        ctx = strip_thinking_tokens(kept)
        next_prompt = prompt_ids + ctx + splice + think_prefix
        input_ids = torch.tensor([next_prompt], device=_device)
        attention_mask = torch.ones_like(input_ids)

        with _lock:
            with torch.no_grad():
                outputs = _model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    do_sample=True,
                    pad_token_id=pad_id,
                )

        new_tokens = outputs[0, len(next_prompt):].tolist()
        while new_tokens and new_tokens[-1] == pad_id:
            new_tokens.pop()

        new_tokens_with_prefix = list(think_prefix) + new_tokens
        new_tokens_start = len(completion_ids)
        completion_ids = completion_ids + new_tokens_with_prefix
        tool_mask = tool_mask + [1] * len(new_tokens_with_prefix)

    # Compute behavior logprobs (for off-policy correction)
    behavior_logprobs = _compute_logprobs(prompt_ids, completion_ids, tool_mask)

    gen_time = time.time() - t0

    result = RolloutResult(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        tool_mask=tool_mask,
        completions=completions_msgs,
        snippet_store=snippet_store,
        tool_call_count=tool_call_count,
        behavior_logprobs=behavior_logprobs,
        metadata=request.metadata,
        batch_id=request.batch_id,
        generation_time=gen_time,
    )
    _result_queue.append(result)
    _result_event.set()

    logger.info(f"Rollout done: {tool_call_count} tools, "
                 f"{len(completion_ids)} tokens, {gen_time:.1f}s "
                 f"(queue: {len(_prompt_queue)} pending, {len(_result_queue)} ready)")


def _compute_logprobs(prompt_ids: list[int], completion_ids: list[int],
                      tool_mask: list[int]) -> list[float] | None:
    """Compute log P(token) under the current (behavior) policy.

    Only computes for model-generated tokens (tool_mask=1).
    Returns None if computation fails.
    """
    try:
        full_ids = prompt_ids + completion_ids
        input_ids = torch.tensor([full_ids], device=_device)

        with _lock:
            with torch.no_grad():
                outputs = _model(input_ids)

        logits = outputs.logits[0]  # [seq_len, vocab]
        # Shift: logits[i] predicts token[i+1]
        prompt_len = len(prompt_ids)
        completion_logprobs = []

        for i, tid in enumerate(completion_ids):
            pos = prompt_len + i - 1  # logits position that predicts this token
            if pos < 0:
                completion_logprobs.append(0.0)
                continue
            log_probs = torch.log_softmax(logits[pos], dim=-1)
            completion_logprobs.append(log_probs[tid].item())

        return completion_logprobs
    except Exception as e:
        logger.warning(f"Failed to compute behavior logprobs: {e}")
        return None


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str, lora_path: str | None = None):
    global _model, _tokenizer, _base_model, _device
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    _device = "cuda"

    logger.info(f"Loading tokenizer: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    logger.info(f"Loading model: {model_name}")
    _base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading LoRA: {lora_path}")
        _model = PeftModel.from_pretrained(_base_model, lora_path)
    else:
        _model = _base_model

    _model.eval()
    logger.info("Model loaded")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-tool-iterations", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of actor threads (1 for single GPU)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    _gen_config.update({
        "max_new_tokens": args.max_new_tokens,
        "max_tool_iterations": args.max_tool_iterations,
        "temperature": args.temperature,
    })

    load_model(args.model, args.lora_path)

    # Start actor workers
    for i in range(args.num_workers):
        t = threading.Thread(target=_actor_worker, daemon=True, name=f"actor-{i}")
        t.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
