"""TiToGRPOTrainer — single-GPU trainer with token-space tool calling.

Overrides TRL's _tool_call_loop to splice tool results in token space
instead of re-templating (TI/TO approach from SID-1). Everything runs
on one GPU — no inference server needed.

Also strips <think>...</think> tokens from the carried-forward context
between tool rounds to prevent context blowup.
"""

import json
import logging
import time

import torch
from trl import GRPOTrainer

from src.training.tito import (
    _init_token_ids, _find_tool_call, _parse_tool_call_json,
    _encode_tool_result, strip_thinking_tokens,
    _TOOL_CALL_END_ID,
)

logger = logging.getLogger(__name__)


class TiToGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with token-space tool calling (TI/TO).

    Single-GPU: generation and training happen on the same device.
    Only _tool_call_loop is overridden — generation uses TRL's default.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _init_token_ids(self.processing_class)
        logger.info("TiToGRPOTrainer: TI/TO tool call loop enabled")

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions,
                        logprobs, images, multimodal_fields):
        """Token-space tool calling loop.

        Instead of TRL's default (decode → re-template → re-tokenize),
        we parse tool calls from tokens, execute tools, and splice
        results directly in token space.
        """
        tokenizer = self.processing_class
        device = next(self.model.parameters()).device
        tool_call_count = 0
        tool_failure_count = 0

        # Build tool_mask: 1 = model-generated, 0 = tool splice
        tool_mask_list = [[1] * len(cids) for cids in completion_ids]

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

            logger.debug(f"TI/TO iteration {iteration+1}: {len(idxs_with_tool)} completions have tool calls")

            # Execute tools and splice results
            new_prompt_ids_for_gen = []  # prompts for next generation (context without thinking)
            new_prompt_ids_raw = []     # for padding/batching

            for i, idx in enumerate(idxs_with_tool):
                name, args = tool_calls_parsed[i]
                tool_call_count += 1

                # Dispatch tool via TRL's tool infrastructure
                result = self._dispatch_tito_tool(name, args)

                # Find </tool_call> position
                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                tc_end_pos = None
                for j in range(len(cids) - 1, -1, -1):
                    if cids[j] == _TOOL_CALL_END_ID:
                        tc_end_pos = j + 1
                        break
                if tc_end_pos is None:
                    tc_end_pos = len(cids)

                kept_completion = cids[:tc_end_pos]
                splice_ids = _encode_tool_result(tokenizer, result)

                # Update tool mask
                tool_mask_list[idx] = tool_mask_list[idx][:tc_end_pos] + [0] * len(splice_ids)

                # Update completion_ids (full, for training)
                completion_ids[idx] = kept_completion + splice_ids

                # Build context for next generation (strip thinking)
                pid = prompt_ids[idx] if isinstance(prompt_ids[idx], list) else prompt_ids[idx].tolist()
                ctx_completion = strip_thinking_tokens(kept_completion)
                new_prompt = pid + ctx_completion + splice_ids
                new_prompt_ids_for_gen.append(new_prompt)

            # Batch generate next turn
            if new_prompt_ids_for_gen:
                t0 = time.time()
                # Pad prompts for batched generation
                max_len = max(len(p) for p in new_prompt_ids_for_gen)
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

                padded = []
                attn_masks = []
                for p in new_prompt_ids_for_gen:
                    padding = [pad_id] * (max_len - len(p))
                    padded.append(padding + p)  # left-pad
                    attn_masks.append([0] * len(padding) + [1] * len(p))

                input_ids = torch.tensor(padded, device=device)
                attention_mask = torch.tensor(attn_masks, device=device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_completion_length,
                        temperature=self.args.temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=pad_id,
                    )

                gen_time = time.time() - t0

                # Extract new tokens and append
                for i, idx in enumerate(idxs_with_tool):
                    prompt_len = len(new_prompt_ids_for_gen[i])
                    pad_len = max_len - prompt_len
                    new_tokens = outputs[i, max_len:].tolist()

                    # Strip trailing pad tokens
                    while new_tokens and new_tokens[-1] == pad_id:
                        new_tokens.pop()

                    completion_ids[idx] = completion_ids[idx] + new_tokens
                    tool_mask_list[idx] = tool_mask_list[idx] + [1] * len(new_tokens)

                    # Update completions (message format for reward functions)
                    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    completions[idx].append({"role": "assistant", "content": new_text})

                logger.info(f"TI/TO round {iteration+1}: {len(idxs_with_tool)} continuations in {gen_time:.1f}s")

        # TRL 1.0 expects 6 return values
        return tool_mask_list, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _dispatch_tito_tool(self, name: str, args: dict) -> str:
        """Dispatch a tool call using TRL's registered tools."""
        # TRL stores tool callables in self.tools (list of functions)
        if self.tools:
            for tool in self.tools:
                tool_name = getattr(tool, '__name__', None)
                if tool_name == name:
                    try:
                        return str(tool(**args))
                    except Exception as e:
                        return f"Error: {e}"

        # Fallback: create fresh environment
        from src.env.search_env_v2 import SearchEnvironmentV2
        env = SearchEnvironmentV2()
        if name == "search":
            return env.search(**args)
        elif name == "read":
            return env.read(**args)
        elif name == "submit_ranking":
            return env.submit_ranking(**args)
        return f"Unknown tool: {name}"
