"""Format reward — checks the model produces valid tool calls and a ranking.

Two checks:
1. Tool calls present (model actually used tools)
2. Final response includes a RANKING: line with valid snippet IDs

Returns 1.0 if both, 0.5 if only tools used, 0.0 if no tools at all.
"""

import re


def format_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    rewards = []
    for completion in completions:
        has_tools = False
        has_ranking = False

        for msg in completion:
            if msg.get("role") != "assistant":
                continue

            # Check tool calls (TRL OpenAI format)
            if msg.get("tool_calls"):
                has_tools = True

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Check for RANKING line in final response
            if "RANKING:" in content.upper():
                ids = re.findall(r'[SR]\d+', content)
                if ids:
                    has_ranking = True

        if has_tools and has_ranking:
            rewards.append(1.0)
        elif has_tools:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards
