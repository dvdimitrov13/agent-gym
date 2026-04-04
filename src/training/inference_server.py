#!/usr/bin/env python3
"""Lightweight inference server using plain transformers.

Runs on GPU 0, serves model.generate() over HTTP.
No vLLM — just transformers + FastAPI.

Endpoints:
  POST /generate     — generate completions from token IDs
  POST /update_weights — reload LoRA adapter from disk
  GET  /health       — health check

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m src.training.inference_server \
      --model Qwen/Qwen3-14B --port 8000
"""

import argparse
import logging
import os
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)
app = FastAPI()

# Global state
_model = None
_tokenizer = None
_base_model = None
_lora_path = None
_device = None


class GenerateRequest(BaseModel):
    prompt_ids: list[list[int]]
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class GenerateResponse(BaseModel):
    completion_ids: list[list[int]]
    generation_time: float


class UpdateWeightsRequest(BaseModel):
    lora_path: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/v1/models")
def list_models():
    """Compatibility endpoint for TRL's vLLM client check."""
    return {"data": [{"id": "model", "object": "model"}]}


@app.post("/generate")
def generate(req: GenerateRequest):
    global _model, _tokenizer
    t0 = time.time()

    # Pad prompts to same length (left-padding)
    max_len = max(len(ids) for ids in req.prompt_ids)
    pad_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id

    padded = []
    attention_masks = []
    for ids in req.prompt_ids:
        padding = [pad_id] * (max_len - len(ids))
        padded.append(padding + ids)
        attention_masks.append([0] * len(padding) + [1] * len(ids))

    input_ids = torch.tensor(padded, device=_device)
    attention_mask = torch.tensor(attention_masks, device=_device)

    with torch.no_grad():
        outputs = _model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            pad_token_id=pad_id,
        )

    # Extract only the generated tokens (strip prompt)
    completion_ids = []
    for i, output in enumerate(outputs):
        prompt_len = len(req.prompt_ids[i])
        # Account for left-padding
        pad_len = max_len - prompt_len
        gen_ids = output[max_len:].tolist()
        completion_ids.append(gen_ids)

    gen_time = time.time() - t0
    logger.info(f"Generated {len(completion_ids)} completions, "
                f"avg len={sum(len(c) for c in completion_ids)/len(completion_ids):.0f} "
                f"in {gen_time:.1f}s")

    return GenerateResponse(completion_ids=completion_ids, generation_time=gen_time)


@app.post("/update_weights")
def update_weights(req: UpdateWeightsRequest):
    global _model, _base_model, _lora_path
    t0 = time.time()

    if not os.path.exists(req.lora_path):
        return {"status": "error", "message": f"Path not found: {req.lora_path}"}

    try:
        # Free old model memory before loading new weights
        del _model
        torch.cuda.empty_cache()

        # Reload LoRA adapter
        _model = PeftModel.from_pretrained(_base_model, req.lora_path)
        _model.eval()
        _lora_path = req.lora_path

        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Reloaded LoRA from {req.lora_path} in {time.time()-t0:.1f}s (GPU mem: {mem:.1f}GB)")
        return {"status": "ok", "lora_path": req.lora_path}
    except Exception as e:
        logger.error(f"Failed to reload LoRA: {e}")
        # Try to recover by using base model
        _model = _base_model
        torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}


def load_model(model_name: str, lora_path: str | None = None):
    global _model, _tokenizer, _base_model, _lora_path, _device
    _device = "cuda"

    logger.info(f"Loading tokenizer: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    logger.info(f"Loading model: {model_name}")
    _base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter: {lora_path}")
        _model = PeftModel.from_pretrained(_base_model, lora_path)
    else:
        _model = _base_model

    _model.eval()
    logger.info("Model loaded and ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    load_model(args.model, args.lora_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
