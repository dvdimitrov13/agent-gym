"""TiToGRPOTrainer — single-GPU trainer with token-space tool calling.

Overrides TRL's _tool_call_loop to splice tool results in token space
instead of re-templating (TI/TO approach from SID-1).

Key behaviors:
  - submit_answer immediately terminates a trajectory (no extra generation)
  - No tool call = trajectory terminated (agent must always use tools)
  - Final-round forcing: injects "you must submit" when time runs out
  - force_submit disabled after N steps so model learns to submit on its own
  - Thinking tokens stripped from carried-forward context
"""

import json
import logging
import os
import time

import torch
from trl import GRPOTrainer
from transformers import TrainerCallback

from src.training.tito import (
    _init_token_ids, _find_tool_call, _parse_tool_call_json,
    _encode_tool_result, strip_thinking_tokens,
    _TOOL_CALL_END_ID,
)

logger = logging.getLogger(__name__)

_FINAL_ROUND_SUFFIX = None


def _find_tc_end(cids: list[int]) -> int:
    """Find position after </tool_call> in token list."""
    for j in range(len(cids) - 1, -1, -1):
        if cids[j] == _TOOL_CALL_END_ID:
            return j + 1
    return len(cids)


class TiToGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with token-space tool calling (TI/TO)."""

    def __init__(self, *args, force_submit_until_step: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        _init_token_ids(self.processing_class)
        self.force_submit_until_step = force_submit_until_step
        self._current_step = 0

        global _FINAL_ROUND_SUFFIX
        final_msg = (
            "\n<|im_end|>\n<|im_start|>system\n"
            "Final round: search and read are no longer available. "
            "You must now call submit_answer with your ranked passage IDs.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        _FINAL_ROUND_SUFFIX = self.processing_class.encode(final_msg, add_special_tokens=False)
        logger.info(f"TiToGRPOTrainer: TI/TO enabled, force_submit until step {force_submit_until_step}")

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions,
                        logprobs, images, multimodal_fields):
        """Token-space tool calling loop.

        Rules:
          - submit_answer → execute, STOP trajectory immediately
          - search/read → execute, splice result, generate next turn
          - no tool call → STOP trajectory (agent must always use tools)
          - final iteration + force_submit → inject "must submit" message
        """
        tokenizer = self.processing_class
        device = next(self.model.parameters()).device
        tool_call_count = 0
        tool_failure_count = 0

        tool_mask_list = [[1] * len(cids) for cids in completion_ids]

        force_submit = self._current_step < self.force_submit_until_step
        self._current_step += 1

        # Track active completions (not yet submitted or terminated)
        active = set(range(len(completion_ids)))

        for iteration in range(self.max_tool_calling_iterations):
            if not active:
                break

            is_last_iter = (iteration == self.max_tool_calling_iterations - 1)
            do_force = is_last_iter and force_submit

            # Classify each active completion's tool call
            submitted = []      # (idx, name, args) — called submit_answer
            searching = []      # (idx, name, args) — called search/read
            no_tool = []        # idx — no tool call, terminate

            for idx in list(active):
                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                span = _find_tool_call(cids)
                if not span:
                    no_tool.append(idx)
                    continue
                parsed = _parse_tool_call_json(tokenizer, cids, span[0], span[1])
                if not parsed:
                    no_tool.append(idx)
                    continue
                name, args = parsed
                if name == "submit_answer":
                    submitted.append((idx, name, args))
                else:
                    searching.append((idx, name, args))

            # 1. Handle submit_answer — execute and STOP (no further generation)
            for idx, name, args in submitted:
                tool_call_count += 1
                result = self._dispatch_tito_tool(name, args)
                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                tc_end = _find_tc_end(cids)
                kept = cids[:tc_end]
                splice = _encode_tool_result(tokenizer, result)
                tool_mask_list[idx] = tool_mask_list[idx][:tc_end] + [0] * len(splice)
                completion_ids[idx] = kept + splice
                active.discard(idx)
                logger.debug(f"TI/TO: completion {idx} submitted ranking")

            # 2. Handle no-tool — terminate immediately
            for idx in no_tool:
                active.discard(idx)

            if not active:
                break

            # 3. Handle search/read — execute, splice, prepare next generation
            gen_prompts = []
            gen_idxs = []

            for idx, name, args in searching:
                if idx not in active:
                    continue

                # On force round, don't execute search/read — force submit below
                if do_force:
                    continue

                tool_call_count += 1
                result = self._dispatch_tito_tool(name, args)

                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                tc_end = _find_tc_end(cids)
                kept = cids[:tc_end]
                splice = _encode_tool_result(tokenizer, result)

                tool_mask_list[idx] = tool_mask_list[idx][:tc_end] + [0] * len(splice)
                completion_ids[idx] = kept + splice

                pid = prompt_ids[idx] if isinstance(prompt_ids[idx], list) else prompt_ids[idx].tolist()
                ctx = strip_thinking_tokens(kept)
                gen_prompts.append(pid + ctx + splice)
                gen_idxs.append(idx)

            # 4. Force submit on remaining active completions if last iteration
            if do_force:
                for idx in list(active):
                    cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                    pid = prompt_ids[idx] if isinstance(prompt_ids[idx], list) else prompt_ids[idx].tolist()
                    ctx = strip_thinking_tokens(cids)
                    prompt = pid + ctx + list(_FINAL_ROUND_SUFFIX)
                    gen_prompts.append(prompt)
                    gen_idxs.append(idx)
                    tool_mask_list[idx] = tool_mask_list[idx] + [0] * len(_FINAL_ROUND_SUFFIX)
                    completion_ids[idx] = cids + list(_FINAL_ROUND_SUFFIX)

            # 5. Batch generate next turn
            #    For forced rounds in first 50 steps, suppress <think> so model
            #    goes straight to tool call instead of burning tokens on thinking
            suppress_think = do_force and self._current_step <= 50
            if gen_prompts:
                new_tokens_list = self._batch_generate(
                    gen_prompts, device, tokenizer,
                    suppress_think=suppress_think,
                )
                gen_time = time.time()  # logged below

                for i, idx in enumerate(gen_idxs):
                    new_tokens = new_tokens_list[i]
                    completion_ids[idx] = completion_ids[idx] + new_tokens
                    tool_mask_list[idx] = tool_mask_list[idx] + [1] * len(new_tokens)
                    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    completions[idx].append({"role": "assistant", "content": new_text})

                # Log GPU memory for monitoring OOM risk
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                label = "FORCED" if do_force else f"round {iteration+1}"
                logger.info(f"TI/TO {label}: {len(gen_idxs)} continuations "
                            f"(GPU: {mem_used:.1f}GB alloc, {mem_reserved:.1f}GB reserved)")

        n_submitted = len(completion_ids) - len(active)
        logger.info(f"TI/TO done: {n_submitted}/{len(completion_ids)} submitted, {tool_call_count} tool calls")

        return tool_mask_list, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _batch_generate(self, prompts: list[list[int]], device, tokenizer,
                        suppress_think: bool = False) -> list[list[int]]:
        """Batch generate from a list of token ID prompts. Returns new tokens per prompt."""
        from src.training.tito import _THINK_START_ID

        max_len = max(len(p) for p in prompts)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        padded = []
        attn_masks = []
        for p in prompts:
            padding = [pad_id] * (max_len - len(p))
            padded.append(padding + p)
            attn_masks.append([0] * len(padding) + [1] * len(p))

        input_ids = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(attn_masks, device=device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_completion_length,
            temperature=self.args.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pad_id,
        )
        if suppress_think and _THINK_START_ID is not None:
            gen_kwargs["suppress_tokens"] = [_THINK_START_ID]

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        results = []
        for i in range(len(prompts)):
            new_tokens = outputs[i, max_len:].tolist()
            while new_tokens and new_tokens[-1] == pad_id:
                new_tokens.pop()
            results.append(new_tokens)
        return results

    def _dispatch_tito_tool(self, name: str, args: dict) -> str:
        """Dispatch a tool call."""
        if self.tools:
            for tool in self.tools:
                if getattr(tool, '__name__', None) == name:
                    try:
                        return str(tool(**args))
                    except Exception as e:
                        return f"Error: {e}"

        from src.env.search_env_v2 import SearchEnvironmentV2
        env = SearchEnvironmentV2()
        if name == "search":
            return env.search(**args)
        elif name == "read":
            return env.read(**args)
        elif name == "submit_answer":
            return env.submit_answer(**args)
        return f"Unknown tool: {name}"


class TrajectoryLoggingCallback(TrainerCallback):
    """Log a full decoded trajectory every N steps for diagnostics."""

    def __init__(self, every_n_steps: int = 10, log_dir: str = "/root/trajectories"):
        self.every_n_steps = every_n_steps
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return

        log_path = os.path.join(self.log_dir, f"step_{step:04d}.json")
        log_data = {
            "step": step,
            "metrics": {k: str(v) for k, v in (logs or {}).items()},
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Trajectory log saved: {log_path}")
