#!/usr/bin/env python3
"""M6: GRPO training with tool-calling via TRL.

Wires together: model + dataset + rewards + SearchEnvironment.

Usage:
    python -m src.training.train --config src/training/configs/local_debug.yaml
    python -m src.training.train --config src/training/configs/cloud_14b.yaml
"""

import argparse
import os
import json

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.env.search_env import SearchEnvironment
from src.rewards import retrieval_reward, efficiency_reward, thinking_reward
from src.utils.device import get_device, get_dtype


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(path: str) -> Dataset:
    """Load JSONL dataset into HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    dtype = get_dtype()

    # Silence experimental warnings
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['dataset']}")
    print()

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=dtype,
        device_map="auto" if device == "cuda" else device,
    )

    # LoRA
    peft_config = None
    if config.get("use_lora"):
        print("Applying LoRA...", flush=True)
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", "all-linear"),
            lora_dropout=config.get("lora_dropout", 0.05),
            task_type="CAUSAL_LM",
        )

    # Load dataset
    print("Loading dataset...", flush=True)
    dataset = load_dataset(config["dataset"])
    print(f"Dataset size: {len(dataset)} examples")

    # GRPO config
    output_dir = config.get("output_dir", "checkpoints/debug")
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_generations=config.get("num_generations", 2),
        max_completion_length=config.get("max_completion_length", 256),
        max_tool_calling_iterations=config.get("max_tool_calling_iterations", 3),
        loss_type=config.get("loss_type", "dapo"),
        beta=config.get("beta", 0.0),
        temperature=config.get("temperature", 0.7),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", -1),
        logging_steps=config.get("logging_steps", 1),
        save_steps=config.get("save_steps", 100),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        mask_truncated_completions=config.get("mask_truncated_completions", True),
        bf16=config.get("bf16", False) and device == "cuda",
        report_to="none",  # disable wandb for now
    )

    # Reward functions
    reward_funcs = [retrieval_reward, efficiency_reward, thinking_reward]
    reward_weights = [1.0, 0.5, 0.3]
    grpo_config.reward_weights = reward_weights

    print(f"\nReward functions: {[f.__name__ for f in reward_funcs]}")
    print(f"Reward weights: {reward_weights}")

    # Environment factory
    def env_factory():
        return SearchEnvironment()

    # Create trainer
    print("\nCreating GRPOTrainer...", flush=True)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        environment_factory=env_factory,
        peft_config=peft_config,
    )

    # Train
    print("Starting training...\n", flush=True)
    trainer.train()

    # Save
    print(f"\nSaving to {output_dir}...", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
