"""NDCG retrieval reward — scores the model's ranked passages against gold passages.

Inspired by SID-1: instead of binary "answer found or not", we compute NDCG
over the model's retrieved snippets using content similarity to gold passages.

The model returns a ranking of snippet IDs. Each snippet is matched against
gold passages by content similarity (word overlap for v1, embeddings later).

Score breakdown:
  - For each model snippet, compute relevance = max similarity to any gold passage
  - Compute DCG from the relevance scores in the model's ranked order
  - Normalize by IDCG (perfect ranking) → NDCG in [0, 1]

This gives:
  - Partial credit (found some relevant content)
  - Ordering incentive (relevant snippets ranked higher = better score)
  - Diminishing returns (many mediocre searches < few targeted ones)
"""

import math
import re


def _extract_snippet_texts(completion: list[dict]) -> dict[str, str]:
    """Extract snippet ID → text mapping from tool results in the completion.

    Tool results have IDs like [S1], [S2] for search, [R1], [R2] for read.
    """
    snippets = {}
    current_id = None

    for msg in completion:
        # TRL format: role=tool
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            # Parse snippet IDs from the content
            _parse_snippet_content(content, snippets)

    return snippets


def _parse_snippet_content(content: str, snippets: dict):
    """Parse snippet IDs and content from a tool result."""
    lines = content.split("\n")
    current_id = None
    current_text = []

    for line in lines:
        # Match [S1], [R1] etc at start of line
        m = re.match(r'^\[([SR]\d+)\]', line)
        if m:
            # Save previous snippet
            if current_id and current_text:
                snippets[current_id] = " ".join(current_text)
            current_id = m.group(1)
            # Rest of line after the ID is part of the snippet
            rest = line[m.end():].strip()
            current_text = [rest] if rest else []
        elif current_id and line.strip():
            current_text.append(line.strip())
        elif not line.strip():
            # Blank line ends current snippet
            if current_id and current_text:
                snippets[current_id] = " ".join(current_text)
            current_id = None
            current_text = []

    # Save last snippet
    if current_id and current_text:
        snippets[current_id] = " ".join(current_text)


def _extract_model_ranking(completion: list[dict]) -> list[str]:
    """Extract the model's ranked snippet IDs from submit_ranking tool call.

    Supports two formats:
    1. Tool call: submit_ranking(passage_ids=["S3", "R1", "S1"])
    2. Fallback text: RANKING: S3, R1, S1
    """
    import json as _json

    # Primary: look for submit_ranking tool call (TRL OpenAI format)
    for msg in reversed(completion):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            if func.get("name") == "submit_ranking":
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        continue
                ids = args.get("passage_ids", [])
                valid = [s for s in ids if re.match(r'^[SR]\d+$', s)]
                if valid:
                    return valid

    # Fallback: look for RANKING: text in final response
    for msg in reversed(completion):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for line in content.split("\n"):
            if line.strip().upper().startswith("RANKING:"):
                ids_str = line.split(":", 1)[1].strip()
                ids = [s.strip() for s in ids_str.split(",") if s.strip()]
                valid = [s for s in ids if re.match(r'^[SR]\d+$', s)]
                if valid:
                    return valid
    return []


def _word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Compute word overlap similarity between two texts.

    Returns fraction of significant words in text_a that appear in text_b.
    """
    if not text_a or not text_b:
        return 0.0
    words_a = {w.lower() for w in text_a.split() if len(w) > 2}
    if not words_a:
        return 0.0
    text_b_lower = text_b.lower()
    matched = sum(1 for w in words_a if w in text_b_lower)
    return matched / len(words_a)


def _compute_relevance(snippet_text: str, gold_passages: list[dict]) -> float:
    """Compute relevance of a snippet against gold passages.

    Returns max similarity to any gold passage.
    """
    if not gold_passages:
        return 0.0
    max_sim = 0.0
    for gp in gold_passages:
        gold_text = gp.get("content", "")
        sim = _word_overlap_similarity(gold_text, snippet_text)
        max_sim = max(max_sim, sim)
    return max_sim


def _dcg(relevances: list[float]) -> float:
    """Compute Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def _ndcg(relevances: list[float]) -> float:
    """Compute NDCG from a list of relevance scores."""
    if not relevances:
        return 0.0
    dcg = _dcg(relevances)
    # IDCG: best possible ordering
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def ndcg_reward(
    completions: list[list[dict]],
    gold_passages: list[list[dict]] | None = None,
    **kwargs,
) -> list[float]:
    """NDCG-based retrieval reward.

    Compares the model's ranked snippets against gold passages using
    content similarity. Returns NDCG score in [0, 1].

    If gold_passages not provided, falls back to checking if answer
    appears in any tool result (binary, like old retrieval_reward).
    """
    if gold_passages is None:
        return [0.5] * len(completions)  # neutral if no gold data

    rewards = []
    for completion, gold in zip(completions, gold_passages):
        # Extract snippets from tool results
        snippet_texts = _extract_snippet_texts(completion)

        # Extract model's ranking
        model_ranking = _extract_model_ranking(completion)

        if not model_ranking or not snippet_texts:
            # Model didn't produce a ranking — zero reward
            rewards.append(0.0)
            continue

        # Compute relevance for each ranked snippet
        relevances = []
        for sid in model_ranking:
            text = snippet_texts.get(sid, "")
            rel = _compute_relevance(text, gold)
            relevances.append(rel)

        # Also check unranked snippets — if model missed relevant ones, NDCG captures it
        # through the IDCG denominator being higher than achievable DCG
        all_relevances = []
        for sid, text in snippet_texts.items():
            rel = _compute_relevance(text, gold)
            all_relevances.append(rel)

        # Compute NDCG
        if all_relevances:
            # Use full set for IDCG (what the model could have achieved)
            dcg = _dcg(relevances)
            ideal = sorted(all_relevances, reverse=True)[:len(model_ranking)]
            idcg = _dcg(ideal)
            score = dcg / idcg if idcg > 0 else 0.0
        else:
            score = 0.0

        rewards.append(score)

    return rewards
