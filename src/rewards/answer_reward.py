"""Answer correctness reward for GRPO training.

Extracts the model's answer from <answer> tags in the final assistant message,
normalizes it, and compares against ground truth + aliases.

Returns 1.0 for correct, 0.0 for wrong.
"""

import re


def _normalize(s: str) -> str:
    """Lowercase, strip articles, punctuation, initials, and extra whitespace."""
    s = s.lower().strip()
    for article in ["the ", "a ", "an "]:
        if s.startswith(article):
            s = s[len(article):]
    s = re.sub(r"[^\w\s]", "", s)
    # Remove single-character words (initials like "A" in "Henry A. Wallace")
    s = re.sub(r"\b\w\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_answer(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _get_last_assistant_text(completion: list[dict]) -> str:
    """Get the text content of the last assistant message."""
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


def _check_match(predicted: str, ground_truth: str, aliases: list[str] | None = None) -> bool:
    """Check if predicted answer matches any ground truth variant."""
    pred = _normalize(predicted)
    if not pred:
        return False
    targets = [ground_truth] + (aliases or [])
    for target in targets:
        t = _normalize(target)
        if not t:
            continue
        if pred == t or t in pred or pred in t:
            return True
    return False


def answer_reward(
    completions: list[list[dict]],
    answer: list[str],
    answer_aliases: list[list[str]] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 if final answer is correct, 0.0 otherwise."""
    if answer_aliases is None:
        answer_aliases = [[] for _ in answer]

    rewards = []
    for completion, gt, aliases in zip(completions, answer, answer_aliases):
        text = _get_last_assistant_text(completion)
        predicted = _extract_answer(text)
        correct = _check_match(predicted, gt, aliases) if predicted else False
        rewards.append(1.0 if correct else 0.0)
    return rewards
