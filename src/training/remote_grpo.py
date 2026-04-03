"""RemoteGRPOTrainer — subclass that sends generation to an inference server.

Overrides _generate_single_turn to call our FastAPI server on GPU 0
instead of using the local model.generate(). TRL handles everything
else: tool calling loop, rewards, GRPO math, gradient updates.
"""

import logging
import time

import requests
from trl import GRPOTrainer
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

LORA_SYNC_PATH = "/tmp/lora_checkpoint"


class RemoteGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that generates via a remote inference server."""

    def __init__(self, *args, inference_server_url: str = "http://localhost:8000", **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_server_url = inference_server_url

        # Verify server is reachable
        try:
            r = requests.get(f"{self.inference_server_url}/health", timeout=5)
            logger.info(f"Inference server connected: {r.json()}")
        except Exception as e:
            raise ConnectionError(f"Cannot reach inference server at {self.inference_server_url}: {e}")

        # Add weight sync callback
        self.add_callback(_WeightSyncCallback(self.inference_server_url))
        logger.info("Weight sync callback added")

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Override: call remote server instead of local model.generate().

        Returns (completion_ids, logprobs) matching parent signature.
        """
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # prompt_ids is already a list of token ID lists, repeated for num_generations
        # (TRL repeats them before calling this method)
        t0 = time.time()

        response = requests.post(
            f"{self.inference_server_url}/generate",
            json={
                "prompt_ids": prompt_ids,
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
        logger.info(f"Remote gen: {len(completion_ids)} completions in {gen_time:.1f}s "
                     f"(server: {result['generation_time']:.1f}s)")

        logprobs = None  # TRL recomputes from training model
        return completion_ids, logprobs


class _WeightSyncCallback(TrainerCallback):
    """Sync LoRA weights to inference server after each training step."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def on_step_end(self, args, state, control, model=None, **kwargs):
        try:
            if model is not None:
                model.save_pretrained(LORA_SYNC_PATH)
                r = requests.post(
                    f"{self.server_url}/update_weights",
                    json={"lora_path": LORA_SYNC_PATH},
                    timeout=30,
                )
                if r.status_code == 200:
                    logger.info(f"Step {state.global_step}: weights synced")
                else:
                    logger.warning(f"Weight sync failed: {r.text}")
        except Exception as e:
            logger.warning(f"Weight sync error: {e}")
