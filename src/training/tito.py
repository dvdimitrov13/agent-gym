"""Tokens-In/Tokens-Out (TI/TO) generation loop.

Inspired by SID-1: instead of decoding generated tokens to messages,
re-applying chat template, and re-tokenizing (lossy), we stay in token
space throughout the multi-turn tool-calling loop.

The only decode/encode operations are:
  1. Decode the tool call JSON (small, to extract function name + args)
  2. Encode the tool result text (to splice into the token sequence)

The bulk of the conversation stays as original token IDs, avoiding
the byte→token mismatches that SID-1 found cause training instability.

Qwen3 template boundaries:
  Assistant tool call ends with: </tool_call><|im_end|>
  Tool result is wrapped in:    <|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>
  Next assistant turn starts:   <|im_start|>assistant\n
"""

import json
import logging
import re

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Qwen3 special token IDs (set on first call via tokenizer)
_TOOL_CALL_START_ID = None   # <tool_call>
_TOOL_CALL_END_ID = None     # </tool_call>
_IM_END_ID = None            # <|im_end|>
_EOS_ID = None               # eos_token_id

# Pre-computed splice sequences (set on first call)
_TOOL_RESULT_PREFIX_IDS = None  # <|im_end|>\n<|im_start|>user\n<tool_response>\n
_TOOL_RESULT_SUFFIX_IDS = None  # \n</tool_response><|im_end|>\n<|im_start|>assistant\n


def _init_token_ids(tokenizer: PreTrainedTokenizerBase):
    """Initialize special token IDs from the tokenizer (called once)."""
    global _TOOL_CALL_START_ID, _TOOL_CALL_END_ID, _IM_END_ID, _EOS_ID
    global _TOOL_RESULT_PREFIX_IDS, _TOOL_RESULT_SUFFIX_IDS

    if _TOOL_CALL_START_ID is not None:
        return

    _TOOL_CALL_START_ID = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
    _TOOL_CALL_END_ID = tokenizer.encode("</tool_call>", add_special_tokens=False)[0]
    _IM_END_ID = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    _EOS_ID = tokenizer.eos_token_id

    _TOOL_RESULT_PREFIX_IDS = tokenizer.encode(
        "<|im_end|>\n<|im_start|>user\n<tool_response>\n",
        add_special_tokens=False,
    )
    _TOOL_RESULT_SUFFIX_IDS = tokenizer.encode(
        "\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    logger.info(f"TI/TO initialized: tool_call={_TOOL_CALL_START_ID}/{_TOOL_CALL_END_ID}, "
                f"prefix={len(_TOOL_RESULT_PREFIX_IDS)} tokens, suffix={len(_TOOL_RESULT_SUFFIX_IDS)} tokens")


def _find_tool_call(token_ids: list[int]) -> tuple[int, int] | None:
    """Find the last <tool_call>...</tool_call> span in token IDs.

    Returns (start_idx, end_idx) of the tool call content (exclusive of markers),
    or None if no tool call found.
    """
    # Search backwards for the last </tool_call>
    try:
        end_idx = len(token_ids) - 1 - token_ids[::-1].index(_TOOL_CALL_END_ID)
    except ValueError:
        return None

    # Search backwards from end_idx for <tool_call>
    try:
        start_idx = end_idx - 1 - token_ids[:end_idx][::-1].index(_TOOL_CALL_START_ID)
    except ValueError:
        return None

    return (start_idx + 1, end_idx)  # content between markers


def _parse_tool_call_json(tokenizer: PreTrainedTokenizerBase, token_ids: list[int],
                          start: int, end: int) -> tuple[str, dict] | None:
    """Decode and parse the tool call JSON from token IDs.

    This is the ONLY decode operation — just the small JSON portion.
    Returns (function_name, arguments) or None.
    """
    text = tokenizer.decode(token_ids[start:end], skip_special_tokens=False).strip()
    try:
        data = json.loads(text)
        name = data.get("name", "")
        arguments = data.get("arguments", {})
        return (name, arguments)
    except json.JSONDecodeError:
        logger.warning(f"TI/TO: failed to parse tool call JSON: {text[:100]}")
        return None


def _encode_tool_result(tokenizer: PreTrainedTokenizerBase, result_text: str) -> list[int]:
    """Encode tool result text with template splice tokens.

    Returns: prefix_ids + result_ids + suffix_ids
    """
    result_ids = tokenizer.encode(result_text, add_special_tokens=False)
    return _TOOL_RESULT_PREFIX_IDS + result_ids + _TOOL_RESULT_SUFFIX_IDS


def tito_generate_with_tools(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    tool_dispatch_fn,
    max_new_tokens: int = 1024,
    max_tool_iterations: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> tuple[torch.Tensor, int]:
    """Generate with tool calling, staying in token space (TI/TO).

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt_ids: Input token IDs [1, seq_len] on the model's device.
        tool_dispatch_fn: Callable(name, args) -> str that executes tools.
        max_new_tokens: Max tokens per generation call.
        max_tool_iterations: Max tool call rounds.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        do_sample: Whether to sample.

    Returns:
        (completion_ids, tool_call_count): The full completion token IDs
        (everything after the original prompt) and number of tool calls made.
    """
    _init_token_ids(tokenizer)

    device = prompt_ids.device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    current_ids = prompt_ids  # [1, seq_len]
    prompt_len = prompt_ids.shape[1]
    tool_call_count = 0

    for iteration in range(max_tool_iterations + 1):
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=current_ids,
                attention_mask=torch.ones_like(current_ids),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_id,
            )

        # Extract new tokens only
        new_ids = outputs[0, current_ids.shape[1]:].tolist()

        if not new_ids:
            break

        # Check for tool call in new tokens
        tool_span = _find_tool_call(new_ids)

        if tool_span is None:
            # No tool call — final response, append and done
            current_ids = outputs
            break

        # Parse the tool call (small decode)
        parsed = _parse_tool_call_json(tokenizer, new_ids, tool_span[0], tool_span[1])
        if parsed is None:
            current_ids = outputs
            break

        name, args = parsed
        tool_call_count += 1
        logger.debug(f"TI/TO: tool call #{tool_call_count}: {name}({str(args)[:60]})")

        # Execute tool
        try:
            result = tool_dispatch_fn(name, args)
        except Exception as e:
            result = f"Error: {e}"

        # Encode tool result as token IDs (small encode)
        splice_ids = _encode_tool_result(tokenizer, result)

        # Concatenate in token space: current + new_generation + splice
        # Trim new_ids to end after </tool_call><|im_end|>
        # Find the </tool_call> in new_ids and include up to <|im_end|> after it
        tc_end_pos = None
        for i in range(len(new_ids) - 1, -1, -1):
            if new_ids[i] == _TOOL_CALL_END_ID:
                tc_end_pos = i + 1
                break

        if tc_end_pos is None:
            current_ids = outputs
            break

        # Keep tokens up to and including </tool_call>, then splice
        kept_new = new_ids[:tc_end_pos]
        all_ids = current_ids[0].tolist() + kept_new + splice_ids
        current_ids = torch.tensor([all_ids], device=device)

    # Extract completion (everything after original prompt)
    completion_ids = current_ids[0, prompt_len:]
    return completion_ids, tool_call_count


def tito_generate_batch(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids_list: list[list[int]],
    tool_dispatch_fn,
    device: str = "cuda",
    **kwargs,
) -> list[tuple[list[int], int]]:
    """Generate for a batch of prompts, one at a time (no batched tool calling).

    Returns list of (completion_ids, tool_call_count) per prompt.
    """
    results = []
    for prompt_ids in prompt_ids_list:
        input_ids = torch.tensor([prompt_ids], device=device)
        completion, tc_count = tito_generate_with_tools(
            model, tokenizer, input_ids, tool_dispatch_fn, **kwargs,
        )
        results.append((completion.tolist(), tc_count))
    return results
