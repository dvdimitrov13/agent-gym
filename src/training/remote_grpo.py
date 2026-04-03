"""GRPOTrainer subclass that uses a remote inference server for generation.

GPU 0 runs the inference server (inference_server.py).
GPU 1 runs this trainer — handles RL math, tool calling, rewards, gradients.

The trainer overrides _generate_single_turn to call the remote server,
and syncs LoRA weights back to the server after each training step.
"""

import logging
import time

import requests
import torch
from trl import GRPOTrainer

logger = logging.getLogger(__name__)

LORA_SYNC_PATH = "/tmp/lora_checkpoint"


class RemoteGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that generates completions via a remote inference server."""

    def __init__(self, *args, inference_server_url: str = "http://localhost:8000", **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_server_url = inference_server_url
        self._step_count = 0

        # Verify server is reachable
        try:
            r = requests.get(f"{self.inference_server_url}/health", timeout=5)
            logger.info(f"Inference server connected: {r.json()}")
        except Exception as e:
            raise ConnectionError(f"Cannot reach inference server at {self.inference_server_url}: {e}")

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Override: send generation to remote server instead of local model."""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # Repeat prompts for num_generations
        expanded_prompt_ids = []
        for ids in prompt_ids:
            for _ in range(num_generations):
                expanded_prompt_ids.append(ids)

        t0 = time.time()

        # Call remote server
        response = requests.post(
            f"{self.inference_server_url}/generate",
            json={
                "prompt_ids": expanded_prompt_ids,
                "max_new_tokens": self.max_completion_length,
                "temperature": self.args.temperature,
                "do_sample": True,
            },
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()

        completion_ids = result["completion_ids"]
        gen_time = time.time() - t0
        logger.info(f"Remote generation: {len(completion_ids)} completions in {gen_time:.1f}s "
                     f"(server: {result['generation_time']:.1f}s)")

        # TRL expects logprobs — we don't have them from the server.
        # Set to None; TRL handles this for non-vLLM generation.
        logprobs = None

        # Return in the format TRL expects
        extra_fields = {}
        return completion_ids, logprobs, extra_fields

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override: sync LoRA weights to server after each step."""
        loss = super().training_step(model, inputs, num_items_in_batch)

        self._step_count += 1

        # Sync weights every step
        try:
            # Save LoRA adapter
            self.model.save_pretrained(LORA_SYNC_PATH)

            # Tell server to reload
            r = requests.post(
                f"{self.inference_server_url}/update_weights",
                json={"lora_path": LORA_SYNC_PATH},
                timeout=30,
            )
            if r.status_code == 200:
                logger.info(f"Step {self._step_count}: weights synced to server")
            else:
                logger.warning(f"Step {self._step_count}: weight sync failed: {r.text}")
        except Exception as e:
            logger.warning(f"Step {self._step_count}: weight sync error: {e}")

        return loss
