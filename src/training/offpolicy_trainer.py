"""Off-policy TI/TO GRPO Trainer — runs on GPU 1.

Pulls pre-generated rollouts from the rollout server (GPU 0) instead of
generating locally. Computes rewards, fresh logprobs under current policy,
and trains with importance-weighted GRPO/DAPO loss.

OLMo-3 / PipelineRL inspired architecture:
  - Actor (GPU 0): generates rollouts continuously
  - Learner (GPU 1, this file): trains on completed rollouts
  - Weight sync: pushes LoRA to actor after each step
  - Off-policy: behavior logprobs from actor, fresh logprobs from learner

Key differences from TiToGRPOTrainer:
  - No local generation — all rollouts come from the server
  - Computes logprobs locally for the training loss
  - Handles staleness via importance sampling (built into GRPO/DAPO loss)
"""

import json
import logging
import os
import time

import torch
import requests
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback

from src.rewards.thinking_reward import thinking_multiplier
from src.rewards.format_reward import _extract_submit_ids

logger = logging.getLogger(__name__)

LORA_SYNC_PATH = "/tmp/lora_sync"


class OffPolicyTiToTrainer(GRPOTrainer):
    """GRPO trainer that pulls rollouts from a remote rollout server."""

    def __init__(self, *args, server_url: str = "http://localhost:8000",
                 disable_thinking: bool = False, **kwargs):
        kwargs.pop("force_submit_until_step", None)
        kwargs.pop("disable_thinking", None)
        super().__init__(*args, **kwargs)

        self.server_url = server_url
        self.disable_thinking = disable_thinking
        self._snippet_store = {}

        # Verify server
        try:
            r = requests.get(f"{self.server_url}/health", timeout=10)
            info = r.json()
            logger.info(f"Connected to rollout server: {info}")
        except Exception as e:
            raise ConnectionError(f"Cannot reach rollout server at {self.server_url}: {e}")

        # Weight sync callback
        self.add_callback(_WeightSyncCallback(self.server_url))

    def _generate_and_score_completions(self, inputs):
        """Override: pull rollouts from server instead of generating locally.

        This replaces the entire generate→tool_call_loop→score pipeline.
        We submit prompts, fetch completed rollouts, compute rewards locally.
        """
        device = self.accelerator.device

        # Extract prompts from inputs
        prompts = inputs["prompts"]
        prompt_ids = inputs.get("prompt_ids") or [
            self.processing_class.encode(p, add_special_tokens=False) for p in prompts
        ]

        # Build metadata from dataset columns (answer, etc.)
        metadata_list = []
        for i in range(len(prompts)):
            meta = {}
            for key in inputs:
                if key not in ("prompts", "prompt_ids", "input_ids", "attention_mask"):
                    val = inputs[key]
                    if isinstance(val, list) and i < len(val):
                        meta[key] = val[i]
            metadata_list.append(meta)

        num_gens = self.num_generations
        batch_id = int(time.time() * 1000) % 1000000

        # Submit prompts (each prompt generates num_generations rollouts)
        submit_data = []
        for i, pid in enumerate(prompt_ids):
            pid_list = pid if isinstance(pid, list) else pid.tolist()
            for g in range(num_gens):
                submit_data.append({
                    "prompt_ids": pid_list,
                    "prompt_text": prompts[i] if isinstance(prompts[i], str) else "",
                    "metadata": metadata_list[i],
                    "batch_id": batch_id,
                })

        # Update gen config with current temperature
        gen_config = {"temperature": self.args.temperature}

        t0 = time.time()
        r = requests.post(
            f"{self.server_url}/submit_prompts",
            json={"prompts": submit_data, "gen_config": gen_config},
            timeout=30,
        )
        r.raise_for_status()

        # Fetch rollouts
        total_needed = len(submit_data)
        r = requests.post(
            f"{self.server_url}/fetch_rollouts",
            json={"count": total_needed, "timeout": 300},
            timeout=360,
        )
        r.raise_for_status()
        response = r.json()
        rollouts = response["rollouts"]

        fetch_time = time.time() - t0
        logger.info(f"Fetched {len(rollouts)}/{total_needed} rollouts in {fetch_time:.1f}s")

        if len(rollouts) < total_needed:
            logger.warning(f"Only got {len(rollouts)}/{total_needed} rollouts, padding with empty")
            while len(rollouts) < total_needed:
                rollouts.append({
                    "prompt_ids": prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids[0].tolist(),
                    "completion_ids": [],
                    "tool_mask": [],
                    "completions": [],
                    "snippet_store": {},
                    "tool_call_count": 0,
                    "behavior_logprobs": None,
                    "metadata": metadata_list[0],
                    "batch_id": batch_id,
                    "generation_time": 0,
                })

        # Process rollouts into tensors for training
        all_prompt_ids = []
        all_completion_ids = []
        all_tool_masks = []
        all_completions = []  # message format
        all_metadata = []
        snippet_stores = {}

        for i, rollout in enumerate(rollouts):
            all_prompt_ids.append(rollout["prompt_ids"])
            all_completion_ids.append(rollout["completion_ids"])
            all_tool_masks.append(rollout["tool_mask"])
            all_completions.append(rollout["completions"])
            all_metadata.append(rollout["metadata"])
            snippet_stores[i] = rollout["snippet_store"]

        self._snippet_store = snippet_stores

        # Compute logprobs under CURRENT policy (not behavior policy)
        # This is what makes importance sampling work — π(a|s)/π_old(a|s)
        current_logprobs = self._compute_current_logprobs(
            all_prompt_ids, all_completion_ids, device
        )

        # Pad completion_ids to same length
        max_comp_len = max(len(c) for c in all_completion_ids) if all_completion_ids else 0
        pad_id = self.processing_class.pad_token_id or self.processing_class.eos_token_id

        padded_completions = []
        padded_masks = []
        for cids, tmask in zip(all_completion_ids, all_tool_masks):
            pad_len = max_comp_len - len(cids)
            padded_completions.append(cids + [pad_id] * pad_len)
            padded_masks.append(tmask + [0] * pad_len)  # mask padding

        completion_ids_tensor = torch.tensor(padded_completions, device=device)
        tool_mask_tensor = torch.tensor(padded_masks, device=device, dtype=torch.bool)

        # Compute rewards using the same reward functions as TiToGRPOTrainer
        rewards = self._compute_rewards_from_rollouts(
            all_completions, all_metadata, snippet_stores
        )

        # Build the output dict that TRL expects
        # Pad prompt_ids too
        max_prompt_len = max(len(p) for p in all_prompt_ids)
        padded_prompts = []
        for pid in all_prompt_ids:
            pad_len = max_prompt_len - len(pid)
            padded_prompts.append([pad_id] * pad_len + pid)  # left-pad

        prompt_ids_tensor = torch.tensor(padded_prompts, device=device)

        return {
            "prompt_ids": prompt_ids_tensor,
            "prompt_texts": [m.get("prompt_text", "") for m in all_metadata],
            "completion_ids": completion_ids_tensor,
            "completions": all_completions,
            "logprobs": current_logprobs,
            "tool_masks": tool_mask_tensor,
            "rewards": rewards,
        }

    def _compute_current_logprobs(self, prompt_ids_list, completion_ids_list, device):
        """Compute log P(completion | prompt) under current policy.

        These are the 'fresh' logprobs needed for the importance ratio.
        """
        pad_id = self.processing_class.pad_token_id or self.processing_class.eos_token_id
        all_logprobs = []

        # Process one at a time to avoid OOM
        was_training = self.model.training
        self.model.eval()

        for prompt_ids, comp_ids in zip(prompt_ids_list, completion_ids_list):
            if not comp_ids:
                all_logprobs.append([])
                continue

            full_ids = prompt_ids + comp_ids
            input_ids = torch.tensor([full_ids], device=device)

            with torch.no_grad():
                outputs = self.model(input_ids)

            logits = outputs.logits[0]  # [seq_len, vocab]
            prompt_len = len(prompt_ids)

            token_logprobs = []
            for i, tid in enumerate(comp_ids):
                pos = prompt_len + i - 1
                if pos < 0:
                    token_logprobs.append(0.0)
                    continue
                lp = torch.log_softmax(logits[pos], dim=-1)
                token_logprobs.append(lp[tid].item())

            all_logprobs.append(token_logprobs)

        if was_training:
            self.model.train()

        return all_logprobs

    def _compute_rewards_from_rollouts(self, completions_list, metadata_list, snippet_stores):
        """Compute rewards for fetched rollouts.

        Uses the same multiplicative structure: judge × format_scale × thinking_scale
        """
        from src.rewards.llm_judge_reward import llm_judge_reward, set_snippet_store

        set_snippet_store(snippet_stores)

        # Build the arguments for the judge
        # Each "completion" is a list of messages from the rollout
        rewards = []

        # Call judge for all completions
        try:
            judge_scores = llm_judge_reward(
                completions=completions_list,
                **{k: [m.get(k, "") for m in metadata_list]
                   for k in metadata_list[0] if k not in ("prompt_text",)}
            )
        except Exception as e:
            logger.warning(f"Judge failed: {e}, returning zeros")
            judge_scores = [0.0] * len(completions_list)

        # Apply multiplicative scalers
        for i, (completion, score) in enumerate(zip(completions_list, judge_scores)):
            fmt_scale = 1.0
            ids = _extract_submit_ids(completion)
            if ids is not None and i in snippet_stores:
                available = set(snippet_stores[i].keys())
                if ids:
                    valid = sum(1 for pid in ids if pid in available)
                    fmt_scale = valid / len(ids)
                else:
                    fmt_scale = 0.0
            elif ids is None:
                fmt_scale = 0.0

            think_scale = 1.0 if self.disable_thinking else thinking_multiplier(completion)
            rewards.append(score * fmt_scale * think_scale)

        return torch.tensor(rewards, dtype=torch.float32)


class _WeightSyncCallback(TrainerCallback):
    """Push LoRA weights to rollout server after each training step."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        try:
            t0 = time.time()
            model.save_pretrained(LORA_SYNC_PATH)
            r = requests.post(
                f"{self.server_url}/update_weights",
                json={"lora_path": LORA_SYNC_PATH},
                timeout=60,
            )
            elapsed = time.time() - t0
            if r.status_code == 200:
                logger.info(f"Step {state.global_step}: weights synced to server ({elapsed:.1f}s)")
            else:
                logger.warning(f"Weight sync failed: {r.text}")
        except Exception as e:
            logger.warning(f"Weight sync error: {e}")
