"""Retrieval quality reward — did the model find content containing the answer?

Checks ALL tool results (search snippets + read excerpts) for the ground truth
answer. Returns 1.0 if found, 0.0 if not.

This rewards the outcome (found the answer via tools) not the specific path
(which URLs or queries). The efficiency reward separately penalizes wasteful paths.
"""


def _extract_all_tool_results(completion: list[dict]) -> list[str]:
    """Extract the text content of ALL tool results (search snippets + read excerpts)."""
    results = []
    for msg in completion:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                text = block.get("content", "")
                if text:
                    results.append(text)
    return results


def _answer_in_text(answer: str, text: str) -> bool:
    """Check if all significant words of the answer appear in the text."""
    answer_words = set(answer.lower().split())
    # Drop very short words (articles, prepositions)
    answer_words = {w for w in answer_words if len(w) > 2}
    if not answer_words:
        return answer.lower() in text.lower()
    text_lower = text.lower()
    return all(w in text_lower for w in answer_words)


def retrieval_reward(
    completions: list[list[dict]],
    answer: list[str],
    **kwargs,
) -> list[float]:
    """Score retrieval quality: did the model find content containing the answer?

    Returns 1.0 if the ground truth answer appears in any tool result
    (search snippets or read excerpts), 0.0 otherwise.

    This rewards the outcome (found the answer) not the path (which URLs).
    The efficiency reward separately handles whether the path was wasteful.
    """
    rewards = []
    for completion, gt_answer in zip(completions, answer):
        tool_results = _extract_all_tool_results(completion)
        score = 0.0
        for result in tool_results:
            if _answer_in_text(gt_answer, result):
                score = 1.0
                break
        rewards.append(score)
    return rewards
