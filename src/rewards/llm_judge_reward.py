"""LLM-as-a-judge reward — GPT scores retrieval trajectory quality.

Single API call replaces NDCG + efficiency + embedding scoring.
Evaluates: relevance, completeness, source quality, efficiency.

The judge sees the full trajectory: search queries, results, and
the final submitted passages.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_MODEL = "gpt-4.1-mini"

# Snippet store set by TiToGRPOTrainer after each _tool_call_loop.
# Maps: completion_idx → {snippet_id → content}
_snippet_store: dict[int, dict[str, str]] | None = None


def set_snippet_store(store: dict[int, dict[str, str]] | None) -> None:
    """Set the snippet store so the judge can look up submitted passage content.

    Args:
        store: Mapping of completion index to ``{snippet_id: content}``.
    """
    global _snippet_store
    _snippet_store = store


def _get_client() -> OpenAI | None:
    """Lazily initialize the OpenAI client from the environment."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("LLM judge: no OPENAI_API_KEY, returning 0")
            return None
        _client = OpenAI(api_key=api_key)
    return _client


JUDGE_PROMPT = """\
You are scoring a retrieval agent's performance. The agent was asked a question, \
used search tools to find information, then submitted passages as its answer.

Question: {question}

Agent's trajectory:
{trajectory_text}

Submitted passages:
{passages_text}

Score the retrieval from 0 to 10 on each criterion:
- Relevance (0-10): Do the submitted passages contain information needed to answer the question?
- Completeness (0-10): For multi-step questions, are all intermediate facts covered?
- Source quality (0-10): Are the sources authoritative (official sites, major news > blogs/forums)?

If no passages were submitted, all scores are 0.

Respond with ONLY this JSON:
{{"relevance": N, "completeness": N, "source_quality": N}}"""


def _extract_trajectory(
    completion: list[dict[str, Any]], completion_idx: int = -1
) -> tuple[str, str, bool]:
    """Extract a human-readable trajectory summary from a completion.

    Parses tool calls (structured, XML, and raw text formats) and tool
    responses to build a trajectory summary and submitted passages text.

    Args:
        completion: List of message dicts from a single rollout.
        completion_idx: Index into the global snippet store for this completion.

    Returns:
        Tuple of (trajectory_text, passages_text, has_submit).
    """
    trajectory_lines = []
    snippets = {}
    passage_ids = []
    has_submit = False

    for msg in completion:
        role = msg.get("role", "")

        if role == "assistant":
            # Extract tool calls from structured format
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if name == "submit_answer":
                    has_submit = True
                    passage_ids = args.get("passage_ids", []) if isinstance(args, dict) else args
                    trajectory_lines.append(f"→ submit_answer({passage_ids})")
                elif name == "search":
                    query = args.get("query", "?")
                    trajectory_lines.append(f'→ search("{query}")')
                elif name == "read":
                    url = args.get("url", "?")[:60]
                    trajectory_lines.append(f"→ read({url})")

            # Also parse <tool_call> blocks from raw text (TRL passes decoded text)
            content = msg.get("content", "")
            if isinstance(content, str) and "<tool_call>" in content:
                for tc_match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL):
                    try:
                        tc_data = json.loads(tc_match.group(1))
                        name = tc_data.get("name", "")
                        args = tc_data.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)

                        if name == "submit_answer" and not has_submit:
                            has_submit = True
                            passage_ids = args.get("passage_ids", []) if isinstance(args, dict) else args
                            trajectory_lines.append(f"→ submit_answer({passage_ids})")
                        elif name == "search":
                            query = args.get("query", "?")
                            trajectory_lines.append(f'→ search("{query}")')
                        elif name == "read":
                            url = args.get("url", "?")[:60]
                            trajectory_lines.append(f"→ read({url})")
                    except (json.JSONDecodeError, TypeError):
                        pass

        elif role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str):
                _parse_snippets(content, snippets)

        # Also check assistant content for spliced tool_response blocks (TI/TO)
        if role == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and "<tool_response>" in content:
                for resp_match in re.finditer(r'<tool_response>(.*?)</tool_response>', content, re.DOTALL):
                    _parse_snippets(resp_match.group(1), snippets)
                # Show truncated result
                preview = content[:100].replace("\n", " ")
                trajectory_lines.append(f"  ← {preview}...")

    # Build passages text from submitted IDs
    # First try snippet_store (set by TI/TO trainer), then fall back to parsed snippets
    if passage_ids:
        passages_lines = []
        for pid in passage_ids:
            # Try the global snippet store first
            text = None
            if _snippet_store and completion_idx >= 0 and completion_idx in _snippet_store:
                text = _snippet_store[completion_idx].get(pid)
            if not text:
                text = snippets.get(pid)
            if not text:
                text = "(content not found)"
            passages_lines.append(f"[{pid}] {text[:200]}")
        passages_text = "\n".join(passages_lines)
    else:
        passages_text = "(no passages submitted)"

    trajectory_text = "\n".join(trajectory_lines) if trajectory_lines else "(no actions taken)"

    return trajectory_text, passages_text, has_submit


def _judge_single(client: OpenAI, question: str, trajectory_text: str, passages_text: str) -> float:
    """Score a single retrieval trajectory using the LLM judge.

    Args:
        client: OpenAI client instance.
        question: The user's original question.
        trajectory_text: Human-readable summary of search actions taken.
        passages_text: Content of submitted passages.

    Returns:
        Weighted score in [0, 1]: 50% relevance + 30% completeness + 20% source quality.
    """
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            max_completion_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question,
                trajectory_text=trajectory_text,
                passages_text=passages_text,
            )}],
        )
        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)

        relevance = float(data.get("relevance", 0)) / 10
        completeness = float(data.get("completeness", 0)) / 10
        source_quality = float(data.get("source_quality", 0)) / 10

        # Weighted combination
        score = 0.5 * relevance + 0.3 * completeness + 0.2 * source_quality
        return score

    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
        return 0.0


def llm_judge_reward(
    completions: list[list[dict[str, Any]]],
    **kwargs: Any,
) -> list[float]:
    """Score retrieval quality using an LLM judge (GPT 4.1 mini).

    For each completion, extracts the full trajectory (search queries, tool
    results) and submitted passages, then asks the judge to score relevance,
    completeness, and source quality. Returns a weighted score in [0, 1].

    This is the primary reward function. Format validity and thinking length
    are applied as multiplicative scalers by TiToGRPOTrainer.

    Args:
        completions: List of rollout message lists, one per generation.
        **kwargs: Dataset columns forwarded by TRL (must include ``prompts``).

    Returns:
        List of reward scores in [0, 1], one per completion.
    """
    client = _get_client()
    if client is None:
        return [0.0] * len(completions)

    # Get prompts from kwargs (TRL forwards dataset columns)
    prompts = kwargs.get("prompts", None)

    rewards = []
    for i, completion in enumerate(completions):
        question = ""
        if prompts and i < len(prompts):
            prompt = prompts[i]
            if isinstance(prompt, list):
                for msg in prompt:
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
                        break
            elif isinstance(prompt, str):
                question = prompt

        trajectory_text, passages_text, has_submit = _extract_trajectory(completion, completion_idx=i)

        if not has_submit or not question:
            rewards.append(0.0)
            continue

        score = _judge_single(client, question, trajectory_text, passages_text)
        rewards.append(score)

    n_scored = sum(1 for r in rewards if r > 0)
    if n_scored > 0:
        avg = sum(rewards) / len(rewards)
        logger.info(f"LLM judge: {n_scored}/{len(rewards)} scored, avg={avg:.3f}")

    return rewards
