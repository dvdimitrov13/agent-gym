#!/usr/bin/env python3
"""M1: Load Qwen3-0.6B and run a simple prompt to verify the model works.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --model Qwen/Qwen3-0.6B --prompt "What is the capital of France?"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.device import get_device, get_dtype


def main():
    parser = argparse.ArgumentParser(description="Run a simple prompt against a model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    dtype = get_dtype()
    print(f"Device: {device} | Dtype: {dtype}")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"Prompt:   {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
