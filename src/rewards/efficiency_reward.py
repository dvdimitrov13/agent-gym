"""Efficiency reward — penalize search count deviation from ideal.

Ideal search count = num_hops (a 3-hop question should take 3 searches).
Score: max(0, 1 - abs(search_count - num_hops) / num_hops)
Perfect at exactly num_hops, penalized for more or fewer.

Only counts search calls, ignores read and submit_answer.
Returns 0 if submit_answer wasn't called (gated on submission).
"""

import json
import re


def _count_search_calls(completion: list[dict]) -> int:
    """Count search tool calls in a completion."""
    count = 0
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            if func.get("name") == "search":
                count += 1
        # Also check raw text (TI/TO completions)
        content = msg.get("content", "")
        if isinstance(content, str):
            count += len(re.findall(r'"name"\s*:\s*"search"', content))
    return count


def _has_submit(completion: list[dict]) -> bool:
    """Check if submit_answer was called."""
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            if tc.get("function", {}).get("name") == "submit_answer":
                return True
        content = msg.get("content", "")
        if isinstance(content, str) and "submit_answer" in content:
            return True
    return False


def efficiency_reward(
    completions: list[list[dict]],
    num_hops: list[int] | None = None,
    **kwargs,
) -> list[float]:
    """Score efficiency: how close is search count to ideal (num_hops)?

    Returns max(0, 1 - abs(searches - num_hops) / num_hops).
    Gated on submit_answer — no credit if model didn't submit.
    """
    if num_hops is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, ideal in zip(completions, num_hops):
        if not _has_submit(completion):
            rewards.append(0.0)
            continue

        ideal = int(ideal)
        if ideal <= 0:
            rewards.append(1.0)
            continue

        search_count = _count_search_calls(completion)
        if search_count == 0:
            rewards.append(0.0)
            continue

        deviation = abs(search_count - ideal)
        score = max(0.0, 1.0 - deviation / ideal)
        rewards.append(score)

    return rewards
