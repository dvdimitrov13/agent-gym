"""Efficiency reward — penalize trajectories that use more tool calls than the reference.

The reference trajectory (from Sonnet) represents the ideal number of steps.
If the model takes more steps, it gets penalized proportionally.

Score: max(0, 1 - extra_steps / gold_steps)
- Same or fewer steps → 1.0
- Double the steps → 0.0
"""


def _count_tool_calls(completion: list[dict], exclude: set[str] | None = None) -> int:
    """Count tool calls in a completion, optionally excluding certain tools.

    Supports TRL format (tool_calls key on assistant messages)
    and Anthropic format (type=tool_use content blocks).
    """
    exclude = exclude or set()
    count = 0
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        # TRL / OpenAI format: tool_calls key on the message
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            import json as _json
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                if name not in exclude:
                    count += 1
            continue
        # Anthropic format: type=tool_use blocks in content list
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name not in exclude:
                        count += 1
    return count


def efficiency_reward(
    completions: list[list[dict]],
    gold_tool_count: list[int] | None = None,
    **kwargs,
) -> list[float]:
    """Penalize excess tool calls beyond the reference trajectory.

    Returns max(0, 1 - extra_steps / gold_steps).
    If gold_tool_count is not provided, returns 1.0 for all (no penalty).
    """
    if gold_tool_count is None:
        return [1.0] * len(completions)

    rewards = []
    for completion, gold_count in zip(completions, gold_tool_count):
        # Exclude submit_answer from count — it's a terminal action, not a search step
        model_count = _count_tool_calls(completion, exclude={"submit_answer"})

        if gold_count <= 0:
            # No reference — no penalty
            rewards.append(1.0)
        elif model_count == 0:
            # Model didn't use tools at all when it should have
            rewards.append(0.0)
        else:
            extra = max(0, model_count - gold_count)
            score = max(0.0, 1.0 - extra / gold_count)
            rewards.append(score)
    return rewards
