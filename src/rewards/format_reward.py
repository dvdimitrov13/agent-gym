"""Format reward — did the trajectory end with submit_answer?

Bootstrap signal: 1.0 if submit_answer was called, 0.0 otherwise.
Creates gradient toward submitting before NDCG can provide signal.
"""


def format_reward(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    rewards = []
    for completion in completions:
        has_submit = False
        for msg in completion:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                if tc.get("function", {}).get("name") == "submit_answer":
                    has_submit = True
            content = msg.get("content", "")
            if isinstance(content, str) and "submit_answer" in content:
                has_submit = True
        rewards.append(1.0 if has_submit else 0.0)
    return rewards
