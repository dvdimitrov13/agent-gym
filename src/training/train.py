#!/usr/bin/env python3
"""M6: GRPO training with tool-calling via TRL.

Wires together: model + dataset + rewards + SearchEnvironment.
Supports vLLM colocate mode for fast parallel generation.

Usage:
    python -m src.training.train --config src/training/configs/local_debug.yaml
    python -m src.training.train --config src/training/configs/cloud_14b.yaml
"""

import argparse
import logging
import os
import json
import time

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.env.search_env import SearchEnvironment
from src.rewards import retrieval_reward, efficiency_reward, thinking_reward, truncation_reward
from src.utils.device import get_device, get_dtype

logger = logging.getLogger(__name__)


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

    # Setup logging — detailed for debugging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence experimental warnings
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    use_vllm = config.get("use_vllm", False)

    logger.info(f"Device: {device}, dtype: {dtype}")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"vLLM: {use_vllm} (mode: {config.get('vllm_mode', 'n/a')})")
    logger.info(f"max_completion_length: {config.get('max_completion_length')}")
    logger.info(f"max_tool_calling_iterations: {config.get('max_tool_calling_iterations')}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (skip if vLLM handles it)
    model = None
    if not use_vllm:
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            dtype=dtype,
            device_map="auto" if device == "cuda" else device,
        )

    # LoRA
    peft_config = None
    if config.get("use_lora"):
        logger.info(f"LoRA: r={config.get('lora_r')}, alpha={config.get('lora_alpha')}")
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", "all-linear"),
            lora_dropout=config.get("lora_dropout", 0.05),
            task_type="CAUSAL_LM",
        )

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(config["dataset"])
    logger.info(f"Dataset size: {len(dataset)} examples")

    # GRPO config
    output_dir = config.get("output_dir", "checkpoints/debug")
    grpo_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_generations=config.get("num_generations", 2),
        max_completion_length=config.get("max_completion_length", 512),
        max_tool_calling_iterations=config.get("max_tool_calling_iterations", 3),
        loss_type=config.get("loss_type", "dapo"),
        beta=config.get("beta", 0.0),
        temperature=config.get("temperature", 0.7),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", -1),
        logging_steps=config.get("logging_steps", 1),
        save_steps=config.get("save_steps", 10),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        mask_truncated_completions=config.get("mask_truncated_completions", False),
        bf16=config.get("bf16", False) and device == "cuda",
        report_to="none",
    )

    # vLLM settings
    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = config.get("vllm_mode", "colocate")
        grpo_kwargs["vllm_gpu_memory_utilization"] = config.get("vllm_gpu_memory_utilization", 0.7)

    grpo_config = GRPOConfig(**grpo_kwargs)

    # Reward functions
    reward_funcs = [retrieval_reward, efficiency_reward, thinking_reward, truncation_reward]
    reward_weights = [1.0, 0.5, 0.3, 0.3]
    grpo_config.reward_weights = reward_weights

    logger.info(f"Reward functions: {[f.__name__ for f in reward_funcs]}")
    logger.info(f"Reward weights: {reward_weights}")

    # Environment factory
    def env_factory():
        return SearchEnvironment()

    # Create trainer
    logger.info("Creating GRPOTrainer...")
    t0 = time.time()

    trainer_kwargs = dict(
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        environment_factory=env_factory,
        peft_config=peft_config,
    )
    if model is not None:
        trainer_kwargs["model"] = model
    else:
        # vLLM mode — pass model name string
        trainer_kwargs["model"] = config["model_name"]

    trainer = GRPOTrainer(**trainer_kwargs)
    logger.info(f"GRPOTrainer created in {time.time()-t0:.1f}s")

    # Train
    logger.info("Starting training...")
    t0 = time.time()
    trainer.train()
    logger.info(f"Training completed in {time.time()-t0:.1f}s")

    # Save
    logger.info(f"Saving to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
