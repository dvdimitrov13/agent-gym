#!/usr/bin/env python3
"""Off-policy TI/TO training launcher.

Starts the rollout server on GPU 0 and the trainer on GPU 1.

Usage:
  python scripts/train_offpolicy.py --config src/training/configs/offpolicy_0.6b.yaml

Or manually (two terminals):
  CUDA_VISIBLE_DEVICES=0 python -m src.training.rollout_server --model Qwen/Qwen3-0.6B
  CUDA_VISIBLE_DEVICES=1 python scripts/train_offpolicy.py --config ... --server-only
"""

import argparse
import logging
import os
import subprocess
import sys
import time

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig

logger = logging.getLogger(__name__)


def start_server(model_name: str, port: int = 8000, lora_path: str = None,
                 max_new_tokens: int = 512, max_tool_iterations: int = 6,
                 temperature: float = 1.0):
    """Start the rollout server as a subprocess on GPU 0."""
    cmd = [
        sys.executable, "-m", "src.training.rollout_server",
        "--model", model_name,
        "--port", str(port),
        "--max-new-tokens", str(max_new_tokens),
        "--max-tool-iterations", str(max_tool_iterations),
        "--temperature", str(temperature),
    ]
    if lora_path:
        cmd.extend(["--lora-path", lora_path])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = os.getcwd()

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    logger.info(f"Rollout server starting on GPU 0 (pid={proc.pid}, port={port})")

    # Wait for server to be ready
    import requests
    for i in range(60):
        time.sleep(2)
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"Rollout server ready: {r.json()}")
                return proc
        except Exception:
            pass

    # Print server output for debugging
    proc.terminate()
    out = proc.stdout.read().decode()
    logger.error(f"Server failed to start. Output:\n{out[-2000:]}")
    raise RuntimeError("Rollout server failed to start within 120s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--server-only", action="store_true",
                        help="Trainer only — assume server is already running")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    server_port = config.get("server_port", 8000)
    server_url = args.server_url

    # Start server if needed
    server_proc = None
    if not args.server_only:
        server_proc = start_server(
            model_name=model_name,
            port=server_port,
            max_new_tokens=config.get("max_completion_length", 512),
            max_tool_iterations=config.get("max_tool_calling_iterations", 6),
            temperature=config.get("temperature", 1.0),
        )

    try:
        # Set trainer to GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model on GPU 1
        logger.info(f"Loading model on GPU 1: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )

        # LoRA
        if config.get("use_lora", True):
            lora_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                target_modules=config.get("lora_target_modules", "all-linear"),
                lora_dropout=config.get("lora_dropout", 0.05),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            logger.info(f"LoRA applied: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        # Load dataset
        dataset_path = config["dataset"]
        logger.info(f"Loading dataset: {dataset_path}")
        dataset = Dataset.from_json(dataset_path)

        # Reward functions
        from src.rewards.llm_judge_reward import llm_judge_reward
        reward_funcs = [llm_judge_reward]
        reward_weights = [1.0]

        # GRPO config
        grpo_config = GRPOConfig(
            output_dir=config.get("output_dir", "checkpoints/offpolicy"),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
            num_generations=config.get("num_generations", 4),
            max_completion_length=config.get("max_completion_length", 512),
            loss_type=config.get("loss_type", "dapo"),
            beta=config.get("beta", 0.0),
            temperature=config.get("temperature", 1.0),
            learning_rate=config.get("learning_rate", 1e-5),
            num_train_epochs=config.get("num_train_epochs", 1),
            max_steps=config.get("max_steps", 100),
            logging_steps=config.get("logging_steps", 1),
            save_steps=config.get("save_steps", 50),
            bf16=config.get("bf16", True),
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            mask_truncated_completions=False,
        )

        # Create trainer
        from src.training.offpolicy_trainer import OffPolicyTiToTrainer

        trainer = OffPolicyTiToTrainer(
            model=model,
            args=grpo_config,
            processing_class=tokenizer,
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            server_url=server_url,
            disable_thinking=config.get("disable_thinking", True),
        )

        # Train
        logger.info("Starting off-policy training")
        resume_from = config.get("resume_from_checkpoint")
        trainer.train(resume_from_checkpoint=resume_from)
        logger.info("Training complete")

    finally:
        if server_proc:
            logger.info("Stopping rollout server...")
            server_proc.terminate()
            server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
