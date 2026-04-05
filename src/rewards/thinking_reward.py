"""Thinking multiplier — scales other rewards based on thinking length.

Returns a multiplier between 0.5 and 1.0:
  ≤128 tokens: 1.0 (rewards untouched)
  128-256 tokens: linear decay from 1.0 to 0.5
  ≥256 tokens: 0.5 (50% reward reduction)

Used by TiToGRPOTrainer to scale the aggregated reward per rollout.
NOT a standard additive TRL reward — applied as post-processing.
"""

import re


def _count_think_tokens_approx(completion: list[dict]) -> int:
    """Count approximate tokens in <think> blocks (chars ÷ 4)."""
    total_chars = 0
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
            total_chars += len(match.group(1))
    return total_chars // 4


# Current thinking budget — updated by curriculum callback via set_thinking_thresholds
_soft_threshold = 128
_hard_threshold = 256


def set_thinking_thresholds(soft: int, hard: int):
    """Update thresholds to match current curriculum stage."""
    global _soft_threshold, _hard_threshold
    _soft_threshold = soft
    _hard_threshold = hard


def thinking_multiplier(completion: list[dict]) -> float:
    """Compute thinking multiplier for a single completion.

    ≤soft: 1.0
    soft-hard: linear from 1.0 to 0.5
    ≥hard: 0.5
    """
    tokens = _count_think_tokens_approx(completion)
    if tokens <= _soft_threshold:
        return 1.0
    elif tokens >= _hard_threshold:
        return 0.5
    else:
        return 1.0 - 0.5 * (tokens - _soft_threshold) / (_hard_threshold - _soft_threshold)
