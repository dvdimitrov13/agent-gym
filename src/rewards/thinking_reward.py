"""Thinking multiplier — scales other rewards based on thinking length.

Annealing schedule (no hard cutoff — model is free to think longer):
  ≤ target:             1.0 (no penalty)
  target → target+128:  linear 1.0 → 0.5
  target+128 → target+256: linear 0.5 → 0.0
  > target+256:         0.0 (zero reward)

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


# Target thinking length — updated by curriculum callback
_target = 256


def set_thinking_thresholds(soft: int, hard: int):
    """Update target to match current curriculum stage.

    For backwards compatibility, takes soft/hard but only uses soft as target.
    """
    global _target
    _target = soft


def thinking_multiplier(completion: list[dict]) -> float:
    """Compute thinking multiplier for a single completion.

    ≤ target:               1.0
    target → target+128:    linear 1.0 → 0.5
    target+128 → target+256: linear 0.5 → 0.0
    > target+256:           0.0
    """
    tokens = _count_think_tokens_approx(completion)
    if tokens <= _target:
        return 1.0
    elif tokens <= _target + 128:
        return 1.0 - 0.5 * (tokens - _target) / 128
    elif tokens <= _target + 256:
        return 0.5 - 0.5 * (tokens - _target - 128) / 128
    else:
        return 0.0
