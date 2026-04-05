"""NDCG retrieval reward — scores the model's ranked passages against gold passages.

Inspired by SID-1: instead of binary "answer found or not", we compute NDCG
over the model's retrieved snippets using embedding similarity to gold passages.

Uses bge-small-en-v1.5 for fast, lightweight embeddings (~130MB). Gold passage
embeddings are precomputed once; only model snippets are embedded at runtime.

Score breakdown:
  - For each model snippet, compute relevance = max cosine similarity to any gold passage
  - Compute DCG from the relevance scores in the model's ranked order
  - Normalize by IDCG (perfect ranking) → NDCG in [0, 1]
"""

import logging
import math
import re

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model (shared across calls)
_embed_model = None

# Precomputed gold embedding index: list of numpy arrays, one per dataset example
# Set once at training start via set_gold_embedding_index()
_gold_embedding_index = None


_gold_key_to_idx = {}


def _gold_passage_key(passages: list[dict]) -> str:
    """Create a hashable key from gold passages for index lookup."""
    texts = tuple(p.get("content", "")[:100] for p in passages if p.get("content"))
    return str(hash(texts))


def set_gold_embedding_index(embeddings: list, gold_passages_list: list[list[dict]] | None = None):
    """Store precomputed gold embeddings for use during training."""
    global _gold_embedding_index, _gold_key_to_idx
    _gold_embedding_index = embeddings

    # Build key→index mapping if passages provided
    if gold_passages_list:
        _gold_key_to_idx = {}
        for i, passages in enumerate(gold_passages_list):
            if passages and embeddings[i] is not None:
                key = _gold_passage_key(passages)
                _gold_key_to_idx[key] = i

    n = sum(1 for e in embeddings if e is not None)
    logger.info(f"NDCG: gold embedding index set ({n}/{len(embeddings)} examples, {len(_gold_key_to_idx)} keys)")


def _get_embed_model():
    """Lazy-load bge-small-en-v1.5 on first use."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        logger.info(f"NDCG: loaded embedding model (dim={_embed_model.get_sentence_embedding_dimension()})")
    return _embed_model


def precompute_gold_embeddings(gold_passages_list: list[list[dict]]) -> list[np.ndarray | None]:
    """Precompute embeddings for all gold passages in the dataset.

    Call once at training start. Returns a list (one per example) of
    numpy arrays of shape (num_gold_passages, embed_dim), or None if
    no gold passages.
    """
    model = _get_embed_model()
    result = []

    # Batch all gold texts for efficiency
    all_texts = []
    index_map = []  # (example_idx, passage_idx)
    for ex_idx, passages in enumerate(gold_passages_list):
        if not passages:
            result.append(None)
            continue
        for p_idx, p in enumerate(passages):
            text = p.get("content", "")
            if text:
                all_texts.append(text)
                index_map.append((ex_idx, p_idx))
        result.append(None)  # placeholder

    if all_texts:
        embeddings = model.encode(all_texts, batch_size=64, show_progress_bar=False)
        # Distribute back to per-example arrays
        example_embeds = {}
        for (ex_idx, _), emb in zip(index_map, embeddings):
            if ex_idx not in example_embeds:
                example_embeds[ex_idx] = []
            example_embeds[ex_idx].append(emb)
        for ex_idx, embs in example_embeds.items():
            result[ex_idx] = np.stack(embs)

    n_with = sum(1 for r in result if r is not None)
    logger.info(f"NDCG: precomputed gold embeddings for {n_with}/{len(result)} examples "
                f"({len(all_texts)} passages total)")
    return result


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, embed_dim) array."""
    model = _get_embed_model()
    return model.encode(texts, batch_size=32, show_progress_bar=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract_snippet_texts(completion: list[dict]) -> dict[str, str]:
    """Extract snippet ID → text mapping from tool results in the completion."""
    snippets = {}
    for msg in completion:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            _parse_snippet_content(content, snippets)
    return snippets


def _parse_snippet_content(content: str, snippets: dict):
    """Parse snippet IDs and content from a tool result."""
    lines = content.split("\n")
    current_id = None
    current_text = []

    for line in lines:
        m = re.match(r'^\[([SR]\d+)\]', line)
        if m:
            if current_id and current_text:
                snippets[current_id] = " ".join(current_text)
            current_id = m.group(1)
            rest = line[m.end():].strip()
            current_text = [rest] if rest else []
        elif current_id and line.strip():
            current_text.append(line.strip())
        elif not line.strip():
            if current_id and current_text:
                snippets[current_id] = " ".join(current_text)
            current_id = None
            current_text = []

    if current_id and current_text:
        snippets[current_id] = " ".join(current_text)


def _extract_model_ranking(completion: list[dict]) -> list[str]:
    """Extract the model's ranked snippet IDs from submit_answer tool call."""
    import json as _json

    for msg in reversed(completion):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            if func.get("name") == "submit_answer":
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

    # Fallback: RANKING: text
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


RELEVANCE_THRESHOLD = 0.65  # below this cosine similarity, treat as irrelevant


def _compute_relevance_embedding(snippet_emb: np.ndarray, gold_embs: np.ndarray) -> float:
    """Compute max cosine similarity between a snippet and any gold passage.

    Clips to 0 below RELEVANCE_THRESHOLD to filter noise from unrelated content.
    Rescales to [0, 1] above the threshold.
    """
    if gold_embs is None or len(gold_embs) == 0:
        return 0.0
    sims = np.dot(gold_embs, snippet_emb) / (
        np.linalg.norm(gold_embs, axis=1) * np.linalg.norm(snippet_emb) + 1e-8
    )
    max_sim = float(np.max(sims))
    if max_sim < RELEVANCE_THRESHOLD:
        return 0.0
    # Rescale: threshold→0, 1.0→1.0
    return (max_sim - RELEVANCE_THRESHOLD) / (1.0 - RELEVANCE_THRESHOLD)


def _dcg(relevances: list[float]) -> float:
    """Compute Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_reward(
    completions: list[list[dict]],
    gold_passages: list[list[dict]] | None = None,
    _gold_embeddings: list[np.ndarray | None] | None = None,
    **kwargs,
) -> list[float]:
    """NDCG-based retrieval reward using embedding similarity.

    Args:
        completions: Model completions (multi-turn message lists).
        gold_passages: Gold passage dicts with 'content' field.
        _gold_embeddings: Precomputed gold embeddings (optional, for speed).
            If not provided, gold passages are embedded on the fly.
    """
    if gold_passages is None:
        return [0.5] * len(completions)

    rewards = []
    for idx, (completion, gold) in enumerate(zip(completions, gold_passages)):
        snippet_texts = _extract_snippet_texts(completion)
        model_ranking = _extract_model_ranking(completion)

        if not model_ranking or not snippet_texts:
            rewards.append(0.0)
            continue

        # Get gold embeddings: try precomputed index first, then _gold_embeddings arg, then compute
        gold_embs = None
        if _gold_embedding_index is not None:
            # Match by gold passage content hash against the precomputed index
            gold_key = _gold_passage_key(gold)
            if gold_key in _gold_key_to_idx:
                gold_embs = _gold_embedding_index[_gold_key_to_idx[gold_key]]
        if gold_embs is None and _gold_embeddings and idx < len(_gold_embeddings) and _gold_embeddings[idx] is not None:
            gold_embs = _gold_embeddings[idx]
        if gold_embs is None:
            gold_texts = [p.get("content", "") for p in gold if p.get("content")]
            if not gold_texts:
                rewards.append(0.0)
                continue
            gold_embs = _embed_texts(gold_texts)

        # Embed all model snippets in one batch
        ranked_texts = [snippet_texts.get(sid, "") for sid in model_ranking]
        all_snippet_texts = list(snippet_texts.values())

        if not any(ranked_texts):
            rewards.append(0.0)
            continue

        all_texts_to_embed = ranked_texts + [t for t in all_snippet_texts if t not in ranked_texts]
        embeddings = _embed_texts([t for t in all_texts_to_embed if t])

        # Map back to ranked/unranked
        emb_idx = 0
        ranked_relevances = []
        for text in ranked_texts:
            if text:
                rel = _compute_relevance_embedding(embeddings[emb_idx], gold_embs)
                ranked_relevances.append(rel)
                emb_idx += 1
            else:
                ranked_relevances.append(0.0)

        all_relevances = list(ranked_relevances)
        for text in all_snippet_texts:
            if text and text not in ranked_texts:
                rel = _compute_relevance_embedding(embeddings[emb_idx], gold_embs)
                all_relevances.append(rel)
                emb_idx += 1

        # Compute NDCG (ranking quality)
        dcg = _dcg(ranked_relevances)
        ideal = sorted(all_relevances, reverse=True)[:len(model_ranking)]
        idcg = _dcg(ideal)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Mean relevance of ranked items (content quality)
        mean_rel = sum(ranked_relevances) / len(ranked_relevances) if ranked_relevances else 0.0

        # Final score: NDCG × mean_relevance
        # This captures both ranking quality AND content quality.
        # Garbage content ranked well → low score (mean_rel ≈ 0.3)
        # Good content ranked poorly → lower score (ndcg < 1)
        score = ndcg * mean_rel

        rewards.append(score)

    return rewards
