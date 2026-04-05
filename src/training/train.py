#!/usr/bin/env python3
"""M6: GRPO training with tool-calling via TRL.

Supports two modes:
1. Simple: single GPU, model.generate() for rollouts (slow but simple)
2. Pipeline: vLLM server on GPU 0, trainer on GPU 1 (fast, decoupled)

Requires TRL from git main for Qwen3 chat template support.

Usage:
    # Simple mode (single GPU)
    python -m src.training.train --config src/training/configs/cloud_14b.yaml

    # Pipeline mode (2 GPUs — launch via script)
    bash scripts/train_pipeline.sh
"""

import argparse
import logging
import os
import json
import time

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.env.search_env import SearchEnvironment
from src.env.search_env_v2 import SearchEnvironmentV2
from src.rewards import (
    retrieval_reward, efficiency_reward, thinking_reward, truncation_reward,
    ndcg_reward, format_reward,
)
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
    use_vllm = config.get("use_vllm", False)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    logger.info(f"Device: {device}, dtype: {dtype}")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Mode: {'pipeline (vLLM server)' if use_vllm else 'simple (model.generate)'}")
    logger.info(f"max_completion_length: {config.get('max_completion_length')}")
    logger.info(f"max_tool_calling_iterations: {config.get('max_tool_calling_iterations')}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if use_vllm:
        # Pipeline mode: vLLM server handles generation, we just need the model for training
        # Pass model name as string — TRL loads it for training only
        logger.info("Pipeline mode — model loaded by TRL for training, vLLM serves generation")
        model = config["model_name"]
    else:
        # Simple mode: load model for both generation and training
        logger.info("Loading model...")
        load_kwargs = dict(dtype=dtype, attn_implementation="sdpa")
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device
        model = AutoModelForCausalLM.from_pretrained(config["model_name"], **load_kwargs)

        if config.get("torch_compile", False) and device == "cuda":
            logger.info("Applying torch.compile...")
            model = torch.compile(model)

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

    # Precompute gold embeddings for NDCG reward (v2 only)
    if config.get("use_v2_rewards", False):
        from src.rewards.ndcg_reward import precompute_gold_embeddings, set_gold_embedding_index
        gold_passages_list = [ex.get("gold_passages", []) for ex in dataset]
        gold_embs = precompute_gold_embeddings(gold_passages_list)
        set_gold_embedding_index(gold_embs, gold_passages_list)
        logger.info("Gold embeddings precomputed and indexed")

    # GRPO config
    output_dir = config.get("output_dir", "checkpoints/debug")
    grpo_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_generations=config.get("num_generations", 2),
        max_completion_length=config.get("max_completion_length", 1024),
        max_tool_calling_iterations=config.get("max_tool_calling_iterations", 3),
        loss_type=config.get("loss_type", "dapo"),
        beta=config.get("beta", 0.0),
        temperature=config.get("temperature", 0.7),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", -1),
        logging_steps=config.get("logging_steps", 1),
        save_steps=config.get("save_steps", 50),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        mask_truncated_completions=config.get("mask_truncated_completions", False),
        bf16=config.get("bf16", False) and device == "cuda",
        report_to="none",
    )

    # V2 reward/env selection
    use_v2 = config.get("use_v2_rewards", False)
    if use_v2:
        reward_funcs = [ndcg_reward, efficiency_reward, thinking_reward, format_reward]
        grpo_kwargs["reward_weights"] = [1.0, 0.5, 0.3, 0.5]
    else:
        reward_funcs = [retrieval_reward, efficiency_reward, thinking_reward, truncation_reward]
        grpo_kwargs["reward_weights"] = [1.0, 0.5, 0.3, 0.3]

    # vLLM server mode settings
    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = config.get("vllm_mode", "server")
        grpo_kwargs["vllm_server_host"] = config.get("vllm_server_host", "0.0.0.0")
        grpo_kwargs["vllm_server_port"] = config.get("vllm_server_port", 8000)
        grpo_kwargs["vllm_server_timeout"] = config.get("vllm_server_timeout", 300)

    grpo_config = GRPOConfig(**grpo_kwargs)

    logger.info(f"Reward functions: {[f.__name__ for f in reward_funcs]}")
    logger.info(f"Reward weights: {grpo_config.reward_weights}")
    logger.info(f"Zero variance filtering: {config.get('zero_variance_filtering', False)}")

    # Environment factory
    use_v2_env = config.get("use_v2_env", False)
    def env_factory():
        return SearchEnvironmentV2() if use_v2_env else SearchEnvironment()
    logger.info(f"Environment: {'SearchEnvironmentV2 (snippet IDs)' if use_v2_env else 'SearchEnvironment'}")

    # Create trainer
    inference_server = config.get("inference_server_url")
    use_tito = config.get("use_tito", False)
    t0 = time.time()

    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        environment_factory=env_factory,
        peft_config=peft_config,
    )

    if inference_server:
        # Dual-GPU: remote inference server
        from src.training.remote_grpo import RemoteGRPOTrainer
        logger.info(f"Creating RemoteGRPOTrainer (server: {inference_server}, tito: {use_tito})...")
        trainer = RemoteGRPOTrainer(
            **trainer_kwargs,
            inference_server_url=inference_server,
            use_tito=use_tito,
        )
    elif use_tito:
        # Single-GPU with TI/TO: token-space tool calling
        from src.training.tito_trainer import TiToGRPOTrainer, TrajectoryLoggingCallback
        logger.info("Creating TiToGRPOTrainer (single GPU, TI/TO)...")
        trainer = TiToGRPOTrainer(**trainer_kwargs)
        trainer.add_callback(TrajectoryLoggingCallback(every_n_steps=10))
    else:
        # Single-GPU standard: TRL handles everything
        logger.info("Creating GRPOTrainer (local generation)...")
        trainer = GRPOTrainer(**trainer_kwargs)
    logger.info(f"Trainer created in {time.time()-t0:.1f}s")

    # Zero variance filtering: wrap the training step to skip batches with no reward signal
    if config.get("zero_variance_filtering", False):
        _original_training_step = trainer.training_step
        _skipped = [0]

        def _filtered_training_step(model, inputs, num_items_in_batch=None):
            # Check if rewards have any variance in this batch
            if "rewards" in inputs:
                rewards = inputs["rewards"]
                if isinstance(rewards, torch.Tensor) and rewards.std() < 1e-8:
                    _skipped[0] += 1
                    if _skipped[0] % 10 == 1:
                        logger.info(f"Zero variance filtering: skipped {_skipped[0]} batches so far")
                    # Return zero loss to skip this batch
                    return torch.tensor(0.0, device=rewards.device, requires_grad=True)
            return _original_training_step(model, inputs, num_items_in_batch)

        trainer.training_step = _filtered_training_step
        logger.info("Zero variance filtering enabled")

    # Curriculum scheduling (SID-1 style): progressively increase difficulty
    length_stages = config.get("length_schedule")  # e.g. [[0, 256], [100, 512], [200, 1024]]
    tool_iter_stages = config.get("tool_iter_schedule")  # e.g. [[0, 3], [100, 6], [200, 10]]

    if length_stages or tool_iter_stages:
        class _CurriculumCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                step = state.global_step
                if length_stages:
                    current_len = length_stages[0][1]
                    for stage_step, stage_len in length_stages:
                        if step >= stage_step:
                            current_len = stage_len
                    if trainer.max_completion_length != current_len:
                        trainer.max_completion_length = current_len
                        logger.info(f"Curriculum: step {step} → max_completion_length={current_len}")

                if tool_iter_stages:
                    current_iters = tool_iter_stages[0][1]
                    for stage_step, stage_iters in tool_iter_stages:
                        if step >= stage_step:
                            current_iters = stage_iters
                    if trainer.max_tool_calling_iterations != current_iters:
                        trainer.max_tool_calling_iterations = current_iters
                        logger.info(f"Curriculum: step {step} → max_tool_calling_iterations={current_iters}")

        trainer.add_callback(_CurriculumCallback())
        logger.info(f"Curriculum scheduling enabled: length={length_stages}, tool_iters={tool_iter_stages}")

    # Train (resume from checkpoint if specified)
    resume_from = config.get("resume_from_checkpoint")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
    else:
        logger.info("Starting training from scratch...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info(f"Training completed in {time.time()-t0:.1f}s")

    # Save
    logger.info(f"Saving to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
