"""Truncation penalty — penalize completions that hit the token limit.

When a completion reaches max_completion_length without a proper ending,
the model wasted tokens rambling or looping. This reward teaches it
to be concise.

Returns 0.0 if truncated (hit the limit), 1.0 if completed naturally.
"""


def truncation_reward(
    completions: list[list[dict]],
    completion_ids: list[list[int]] | None = None,
    **kwargs,
) -> list[float]:
    """Penalize truncated completions.

    A completion is considered truncated if completion_ids length equals
    max_completion_length (from trainer_state). Falls back to checking
    if the last message has no text content.
    """
    # Get max_completion_length from trainer state if available
    trainer_state = kwargs.get("trainer_state")
    max_len = None
    if trainer_state and hasattr(trainer_state, "max_completion_length"):
        max_len = trainer_state.max_completion_length

    rewards = []
    for i, completion in enumerate(completions):
        # Check via completion_ids length if available
        if completion_ids and i < len(completion_ids) and max_len:
            if len(completion_ids[i]) >= max_len:
                rewards.append(0.0)
                continue

        # Fallback: check if the last assistant message looks complete
        # (has actual text content, not just a partial tool call)
        last_msg = completion[-1] if completion else {}
        if last_msg.get("role") == "assistant":
            content = last_msg.get("content", "")
            if isinstance(content, list):
                has_text = any(
                    b.get("type") == "text" and b.get("text", "").strip()
                    for b in content if isinstance(b, dict)
                )
            else:
                has_text = bool(content.strip())

            if has_text:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards
