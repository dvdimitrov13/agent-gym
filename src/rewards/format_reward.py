"""Format reward — did the model use <answer> tags?

Returns 1.0 if the final assistant message contains <answer>...</answer> tags,
0.0 otherwise. This gives the model a learning signal for output format
even when the answer content is wrong.
"""

import re


def _get_last_assistant_text(completion: list[dict]) -> str:
    for msg in reversed(completion):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                if texts:
                    return " ".join(texts)
    return ""


def format_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    """1.0 if <answer> tags present in final response, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        text = _get_last_assistant_text(completion)
        has_tags = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        rewards.append(1.0 if has_tags else 0.0)
    return rewards
