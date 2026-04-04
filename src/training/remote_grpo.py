"""RemoteGRPOTrainer — subclass that sends generation to an inference server.

Two modes:
1. Standard: overrides _generate_single_turn, TRL handles tool calling loop
2. TI/TO: overrides _generate_single_turn AND _tool_call_loop. Generation is
   still batched via the server, but tool results are spliced in token space
   instead of re-templating (avoids lossy decode→re-encode round-trips).
"""

import json
import logging
import time

import torch
import requests
from trl import GRPOTrainer
from transformers import TrainerCallback

from src.training.tito import (
    _init_token_ids, _find_tool_call, _parse_tool_call_json,
    _encode_tool_result, _TOOL_RESULT_PREFIX_IDS, _TOOL_RESULT_SUFFIX_IDS,
    _TOOL_CALL_END_ID,
)

logger = logging.getLogger(__name__)

LORA_SYNC_PATH = "/tmp/lora_checkpoint"


class RemoteGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that generates via a remote inference server."""

    def __init__(self, *args, inference_server_url: str = "http://localhost:8000",
                 use_tito: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_server_url = inference_server_url
        self.use_tito = use_tito

        if use_tito:
            _init_token_ids(self.processing_class)

        # Verify server is reachable
        try:
            r = requests.get(f"{self.inference_server_url}/health", timeout=5)
            logger.info(f"Inference server connected: {r.json()}")
        except Exception as e:
            raise ConnectionError(f"Cannot reach inference server at {self.inference_server_url}: {e}")

        # Add weight sync callback
        self.add_callback(_WeightSyncCallback(self.inference_server_url))
        logger.info(f"Weight sync callback added (TI/TO: {use_tito})")

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Override: call remote server instead of local model.generate().

        Returns (completion_ids, logprobs) matching parent signature.
        """
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

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions,
                        logprobs, images, multimodal_fields):
        """Override tool call loop.

        In TI/TO mode: parse tool calls from tokens, execute tools, splice
        results in token space, batch-generate next turn via server.
        Same batching as standard, just different splicing.

        In standard mode: fall through to TRL's default.
        """
        if not self.use_tito:
            return super()._tool_call_loop(
                prompts, prompt_ids, completion_ids, completions,
                logprobs, images, multimodal_fields,
            )

        tokenizer = self.processing_class
        tool_call_count = 0
        tool_failure_count = 0
        tool_images = [[] for _ in completions]

        # Build tool_mask: 1 = model-generated, 0 = tool splice
        # Start with all 1s (initial completion is all model-generated)
        tool_mask_list = [[1] * len(cids) for cids in completion_ids]

        # Track full token sequences per completion (prompt + completion so far)
        full_sequences = []
        for pid, cid in zip(prompt_ids, completion_ids):
            if isinstance(pid, list):
                full_sequences.append(pid + cid)
            else:
                full_sequences.append(pid.tolist() + cid)

        # Get environment tools
        env = self._env if hasattr(self, '_env') else None
        if env is None and hasattr(self, 'environment'):
            env = self.environment

        for iteration in range(self.max_tool_calling_iterations):
            # Find which completions have tool calls (in token space)
            idxs_with_tool = []
            tool_calls_parsed = []

            for idx, cids in enumerate(completion_ids):
                if not isinstance(cids, list):
                    cids = cids.tolist()
                span = _find_tool_call(cids)
                if span:
                    parsed = _parse_tool_call_json(tokenizer, cids, span[0], span[1])
                    if parsed:
                        idxs_with_tool.append(idx)
                        tool_calls_parsed.append(parsed)

            if not idxs_with_tool:
                break

            logger.debug(f"TI/TO iteration {iteration+1}: {len(idxs_with_tool)} prompts have tool calls")

            # Execute tools and splice results in token space
            new_prompt_ids = []
            for i, idx in enumerate(idxs_with_tool):
                name, args = tool_calls_parsed[i]
                tool_call_count += 1

                # Dispatch tool
                try:
                    if name == "search":
                        result = self._dispatch_tool(name, args)
                    elif name == "read":
                        result = self._dispatch_tool(name, args)
                    elif name == "submit_ranking":
                        result = self._dispatch_tool(name, args)
                    else:
                        result = f"Unknown tool: {name}"
                except Exception as e:
                    result = f"Error: {e}"
                    tool_failure_count += 1

                # Splice in token space
                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()

                # Find </tool_call> position and trim completion there
                tc_end_pos = None
                for j in range(len(cids) - 1, -1, -1):
                    if cids[j] == _TOOL_CALL_END_ID:
                        tc_end_pos = j + 1
                        break

                if tc_end_pos is None:
                    tc_end_pos = len(cids)

                kept_completion = cids[:tc_end_pos]
                splice_ids = _encode_tool_result(tokenizer, result)

                # Update tool mask: add 0s for splice tokens
                tool_mask_list[idx] = tool_mask_list[idx][:tc_end_pos] + [0] * len(splice_ids)

                # Build new full sequence for next generation
                pid = prompt_ids[idx] if isinstance(prompt_ids[idx], list) else prompt_ids[idx].tolist()
                new_full = pid + kept_completion + splice_ids
                new_prompt_ids.append(new_full)

                # Update completion_ids to include splice
                completion_ids[idx] = kept_completion + splice_ids
                full_sequences[idx] = new_full

            # Batch generate next turn for all prompts with tool calls
            if new_prompt_ids:
                t0 = time.time()
                response = requests.post(
                    f"{self.inference_server_url}/generate",
                    json={
                        "prompt_ids": new_prompt_ids,
                        "max_new_tokens": self.max_completion_length,
                        "temperature": self.args.temperature,
                        "do_sample": True,
                    },
                    timeout=600,
                )
                response.raise_for_status()
                new_completions = response.json()["completion_ids"]
                gen_time = time.time() - t0
                logger.info(f"TI/TO round {iteration+1}: {len(new_completions)} "
                            f"continuations in {gen_time:.1f}s")

                # Append new completion tokens
                for i, idx in enumerate(idxs_with_tool):
                    new_tokens = new_completions[i]
                    completion_ids[idx] = completion_ids[idx] + new_tokens
                    tool_mask_list[idx] = tool_mask_list[idx] + [1] * len(new_tokens)
                    full_sequences[idx] = full_sequences[idx] + new_tokens

                    # Update completions (message format for reward functions)
                    # Decode the new tokens to build completion messages
                    if i < len(new_completions):
                        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        completions[idx].append({"role": "assistant", "content": new_text})

        return tool_mask_list, completions, completion_ids, logprobs, tool_call_count, tool_failure_count, tool_images

    def _dispatch_tool(self, name, args):
        """Dispatch a tool call to the environment."""
        # Access the environment from the trainer's tool dict
        if hasattr(self, 'tools') and self.tools:
            for tool in self.tools:
                if hasattr(tool, '__name__') and tool.__name__ == name:
                    return tool(**args)
                if callable(tool) and getattr(tool, 'name', None) == name:
                    return tool(**args)

        # Fallback: create a fresh environment
        from src.env.search_env_v2 import SearchEnvironmentV2
        env = SearchEnvironmentV2()
        if name == "search":
            return env.search(**args)
        elif name == "read":
            return env.read(**args)
        elif name == "submit_ranking":
            return env.submit_ranking(**args)
        return f"Unknown tool: {name}"


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
                    timeout=60,
                )
                if r.status_code == 200:
                    logger.info(f"Step {state.global_step}: weights synced")
                else:
                    logger.warning(f"Weight sync failed: {r.text}")
        except Exception as e:
            logger.warning(f"Weight sync error: {e}")
