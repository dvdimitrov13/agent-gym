"""Thinking reward — scales other rewards based on thinking length.

Acts as a penalty that decays total reward for verbose thinking:
  ≤128 tokens: 0.0 (no penalty, other rewards at full value)
  128-256 tokens: linear decay from 0.0 to -1.0
  ≥256 tokens: -1.0 (max penalty, ~50% reward reduction)

With other rewards summing to ~2.0 max (judge=1.0 + efficiency=0.5 + format=0.5),
a -1.0 penalty at 256 tokens reduces total from 2.0 to 1.0 (50% reduction).

Measured in tokens by counting characters in <think> blocks (÷4 approximation).
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
    return total_chars // 4  # rough token estimate


def thinking_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    """Penalty for verbose thinking. Returns 0 to -1.

    ≤128 tokens: 0.0 (no penalty)
    128-256 tokens: linear from 0.0 to -1.0
    ≥256 tokens: -1.0
    """
    rewards = []
    for completion in completions:
        tokens = _count_think_tokens_approx(completion)

        if tokens <= 128:
            penalty = 0.0
        elif tokens >= 256:
            penalty = -1.0
        else:
            penalty = -1.0 * (tokens - 128) / (256 - 128)

        rewards.append(penalty)
    return rewards
