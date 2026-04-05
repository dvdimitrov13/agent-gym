"""Thinking reward — encourage brief planning, penalize long rambling.

Qwen3 naturally produces <think>...</think> blocks. We want:
- Short thinking (planning what to search) → 1.0
- Long thinking (reasoning from memory instead of searching) → decays to 0.0
- No thinking → 0.5 (neutral)

Measured in tokens (not words) for precise budget control.
Default: 1.0 if ≤128 tokens, linear decay to 0.0 at 256 tokens.
"""

import re


def _count_think_tokens(completion: list[dict], tokenizer=None) -> int:
    """Count total tokens in <think> blocks across all assistant messages.

    Falls back to word count * 1.3 if no tokenizer provided.
    """
    total_text = ""
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            content = " ".join(texts)
        for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
            total_text += match.group(1).strip() + " "

    if not total_text.strip():
        return 0

    if tokenizer:
        return len(tokenizer.encode(total_text, add_special_tokens=False))
    # Approximate: 1 word ≈ 1.3 tokens
    return int(len(total_text.split()) * 1.3)


def thinking_reward(
    completions: list[list[dict]],
    short_threshold: int = 128,
    long_threshold: int = 256,
    **kwargs,
) -> list[float]:
    """Score thinking length: brief planning good, long rambling bad.

    Args:
        short_threshold: Token count at or below which score is 1.0
        long_threshold: Token count at or above which score is 0.0
    """
    rewards = []
    for completion in completions:
        total_tokens = _count_think_tokens(completion)

        if total_tokens == 0:
            rewards.append(0.5)
            continue

        if total_tokens <= short_threshold:
            score = 1.0
        elif total_tokens >= long_threshold:
            score = 0.0
        else:
            score = 1.0 - (total_tokens - short_threshold) / (long_threshold - short_threshold)

        rewards.append(score)
    return rewards
