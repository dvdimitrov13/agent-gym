"""Format reward — rewards tool use and submit_answer completion.

Bootstrapping signal: teaches the model to always use tools and
end with submit_answer. Becomes redundant once NDCG provides signal.

Score:
  1.0 — submit_answer was called (ideal)
  0.5 — tools were used but no submit (searching at least)
  0.0 — no tools at all (just text response)
"""

import json
import re


def format_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    rewards = []
    for completion in completions:
        has_tools = False
        has_submit = False

        for msg in completion:
            if msg.get("role") != "assistant":
                continue

            # Check TRL OpenAI format: tool_calls key
            for tc in msg.get("tool_calls", []):
                has_tools = True
                func = tc.get("function", {})
                name = func.get("name", "")
                if name == "submit_answer":
                    has_submit = True

            # Check raw text for tool calls (TI/TO completions)
            content = msg.get("content", "")
            if isinstance(content, str) and "<tool_call>" in content:
                has_tools = True
                if "submit_answer" in content:
                    has_submit = True

        if has_submit:
            rewards.append(1.0)
        elif has_tools:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards
