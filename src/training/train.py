#!/usr/bin/env python3
"""M6: GRPO training with tool-calling via Unsloth + TRL.

Uses Unsloth's FastLanguageModel for memory-efficient loading + vLLM standby
mode for fast generation. TRL's GRPOTrainer handles the RL loop and tool calling
via environment_factory.

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
from trl import GRPOConfig, GRPOTrainer

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
    use_unsloth = config.get("use_unsloth", False)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence experimental warnings
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    logger.info(f"Device: {device}, dtype: {dtype}")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Unsloth: {use_unsloth}")
    logger.info(f"max_completion_length: {config.get('max_completion_length')}")
    logger.info(f"max_tool_calling_iterations: {config.get('max_tool_calling_iterations')}")

    # Load model + tokenizer
    peft_config = None

    if use_unsloth:
        # Enable vLLM standby mode BEFORE importing unsloth
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
        from unsloth import FastLanguageModel

        fast_inference = config.get("fast_inference", False)
        logger.info(f"Loading model with Unsloth (fast_inference={fast_inference})...")
        load_kwargs = dict(
            model_name=config["model_name"],
            max_seq_length=config.get("max_seq_length", 4096),
            load_in_4bit=config.get("load_in_4bit", False),
            fast_inference=fast_inference,
        )
        if fast_inference:
            load_kwargs["max_lora_rank"] = config.get("lora_r", 16)
            load_kwargs["gpu_memory_utilization"] = config.get("gpu_memory_utilization", 0.85)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

        # Unsloth applies LoRA via get_peft_model
        logger.info(f"Applying LoRA via Unsloth: r={config.get('lora_r')}, alpha={config.get('lora_alpha')}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=config.get("lora_dropout", 0.05),
        )
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model with transformers...")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            dtype=dtype,
            device_map="auto" if device == "cuda" else device,
            attn_implementation="sdpa",  # scaled dot product attention (flash-like)
        )

        # torch.compile for faster forward pass (CUDA only)
        if device == "cuda" and config.get("torch_compile", True):
            logger.info("Applying torch.compile...")
            model = torch.compile(model)

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

    # vLLM via Unsloth standby (not TRL's use_vllm)
    # Unsloth manages vLLM internally when fast_inference=True

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

    # Requires TRL from git main (pip install git+https://github.com/huggingface/trl.git)
    # which has Qwen3 chat template support (PR #5330)
    #
    # Unsloth modifies the tokenizer's chat_template, causing TRL's exact-match
    # checks to fail. Restore the original Qwen3 template and set response_schema.
    if use_unsloth:
        from transformers import AutoTokenizer as _AT
        original_tok = _AT.from_pretrained(config["model_name"])
        if original_tok.chat_template and original_tok.chat_template != tokenizer.chat_template:
            logger.info("Restoring original Qwen3 chat_template (Unsloth modified it)")
            tokenizer.chat_template = original_tok.chat_template
        from trl.chat_template_utils import qwen3_schema
        logger.info("Setting Qwen3 response_schema")
        tokenizer.response_schema = qwen3_schema
        del original_tok

    # Create trainer
    logger.info("Creating GRPOTrainer...")
    t0 = time.time()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        environment_factory=env_factory,
        peft_config=peft_config,
    )
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
