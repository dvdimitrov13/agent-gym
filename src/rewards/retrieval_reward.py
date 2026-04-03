"""Retrieval quality reward — did the model find the right content?

Two components:
1. URL recall: what fraction of gold URLs (from reference trajectory) did the model find?
2. Content match: does the ground truth answer appear in any read() results?

The gold URLs and gold tool count come from the dataset (extracted from Sonnet's
reference trajectories during data generation).
"""

import re


def _extract_search_urls(completion: list[dict]) -> set[str]:
    """Extract URLs that appeared in search results (tool_result blocks)."""
    urls = set()
    for msg in completion:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = block.get("content", "")
            # Match URLs in search result format: "    https://..."
            for match in re.finditer(r"^\s+(https?://\S+)", text, re.MULTILINE):
                urls.add(match.group(1))
    return urls


def _extract_read_urls(completion: list[dict]) -> set[str]:
    """Extract URLs the model explicitly called read() on."""
    urls = set()
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if (isinstance(block, dict) and block.get("type") == "tool_use"
                    and block.get("name") == "read"):
                url = block.get("input", {}).get("url", "")
                if url:
                    urls.add(url)
    return urls


def _extract_read_results(completion: list[dict]) -> list[str]:
    """Extract the text content of all read() tool results."""
    results = []
    # Track which tool_use IDs are read calls
    read_ids = set()
    for msg in completion:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if (isinstance(block, dict) and block.get("type") == "tool_use"
                        and block.get("name") == "read"):
                    read_ids.add(block.get("id", ""))

    for msg in completion:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if (isinstance(block, dict) and block.get("type") == "tool_result"
                    and block.get("tool_use_id", "") in read_ids):
                results.append(block.get("content", ""))
    return results


def _normalize_url(url: str) -> str:
    """Strip trailing slashes and fragments for comparison."""
    url = url.rstrip("/")
    url = url.split("#")[0]
    return url.lower()


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
    gold_urls: list[list[str]] | None = None,
    **kwargs,
) -> list[float]:
    """Score retrieval quality: URL recall + content match.

    Returns a score between 0.0 and 1.0:
    - 0.5 weight on URL recall (fraction of gold URLs found)
    - 0.5 weight on content match (answer words in read results)

    If gold_urls is not provided, only content match is scored.
    """
    rewards = []
    for i, (completion, gt_answer) in enumerate(zip(completions, answer)):
        found_urls = _extract_search_urls(completion) | _extract_read_urls(completion)
        found_urls_norm = {_normalize_url(u) for u in found_urls}

        # URL recall
        if gold_urls and i < len(gold_urls) and gold_urls[i]:
            gold_norm = {_normalize_url(u) for u in gold_urls[i]}
            overlap = len(found_urls_norm & gold_norm)
            url_score = overlap / len(gold_norm) if gold_norm else 0.0
        else:
            url_score = None

        # Content match: answer words in read results
        read_results = _extract_read_results(completion)
        content_score = 0.0
        if read_results:
            for result in read_results:
                if _answer_in_text(gt_answer, result):
                    content_score = 1.0
                    break

        # Combine
        if url_score is not None:
            reward = 0.5 * url_score + 0.5 * content_score
        else:
            reward = content_score

        rewards.append(reward)
    return rewards
