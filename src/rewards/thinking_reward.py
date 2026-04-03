"""Thinking reward — encourage brief planning, penalize long rambling.

Qwen3 naturally produces <think>...</think> blocks. We want:
- Short thinking (planning what to search) → 1.0
- Long thinking (reasoning from memory instead of searching) → decays to 0.0
- No thinking → 0.5 (neutral)

Score: 1.0 if ≤50 words, linear decay to 0.0 at 200 words.
"""

import re


def _extract_think_blocks(completion: list[dict]) -> list[str]:
    """Extract all <think> blocks from assistant messages."""
    blocks = []
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            content = " ".join(texts)
        for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
            blocks.append(match.group(1).strip())
    return blocks


def thinking_reward(
    completions: list[list[dict]],
    short_threshold: int = 50,
    long_threshold: int = 200,
    **kwargs,
) -> list[float]:
    """Score thinking length: brief planning good, long rambling bad.

    Args:
        short_threshold: Word count at or below which score is 1.0
        long_threshold: Word count at or above which score is 0.0
    """
    rewards = []
    for completion in completions:
        blocks = _extract_think_blocks(completion)

        if not blocks:
            rewards.append(0.5)
            continue

        # Total thinking words across all think blocks
        total_words = sum(len(block.split()) for block in blocks)

        if total_words <= short_threshold:
            score = 1.0
        elif total_words >= long_threshold:
            score = 0.0
        else:
            # Linear decay between thresholds
            score = 1.0 - (total_words - short_threshold) / (long_threshold - short_threshold)

        rewards.append(score)
    return rewards
