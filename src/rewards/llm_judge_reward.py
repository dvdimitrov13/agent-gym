"""LLM-as-a-judge reward — GPT scores retrieval trajectory quality.

Single API call replaces NDCG + efficiency + embedding scoring.
Evaluates: relevance, completeness, source quality, efficiency.

The judge sees the full trajectory: search queries, results, and
the final submitted passages.
"""

import json
import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

_client = None
_MODEL = "gpt-4.1-mini"


def _get_client():
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


def _extract_trajectory(completion: list[dict]) -> tuple[str, str, bool]:
    """Extract trajectory summary, submitted passages text, and whether submit was called.

    Returns (trajectory_text, passages_text, has_submit).
    """
    trajectory_lines = []
    snippets = {}
    passage_ids = []
    has_submit = False

    for msg in completion:
        role = msg.get("role", "")

        if role == "assistant":
            # Extract tool calls
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
                    passage_ids = args.get("passage_ids", [])
                    trajectory_lines.append(f"→ submit_answer({passage_ids})")
                elif name == "search":
                    query = args.get("query", "?")
                    trajectory_lines.append(f"→ search(\"{query}\")")
                elif name == "read":
                    url = args.get("url", "?")[:60]
                    trajectory_lines.append(f"→ read({url})")

            # Check raw text for tool calls too
            content = msg.get("content", "")
            if isinstance(content, str) and "submit_answer" in content:
                match = re.search(r'"passage_ids"\s*:\s*\[(.*?)\]', content)
                if match:
                    ids = re.findall(r'"([SR]\d+)"', match.group(1))
                    if ids and not passage_ids:
                        passage_ids = ids
                        has_submit = True

        elif role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str):
                # Parse snippet IDs
                for line in content.split("\n"):
                    m = re.match(r'^\[([SR]\d+)\](.*)', line)
                    if m:
                        sid = m.group(1)
                        rest = m.group(2).strip()
                        # Get next non-empty line as content
                        snippets[sid] = rest[:200]
                # Show truncated result
                preview = content[:100].replace("\n", " ")
                trajectory_lines.append(f"  ← {preview}...")

    # Build passages text from submitted IDs
    if passage_ids:
        passages_lines = []
        for pid in passage_ids:
            text = snippets.get(pid, "(content not found)")
            passages_lines.append(f"[{pid}] {text[:200]}")
        passages_text = "\n".join(passages_lines)
    else:
        passages_text = "(no passages submitted)"

    trajectory_text = "\n".join(trajectory_lines) if trajectory_lines else "(no actions taken)"

    return trajectory_text, passages_text, has_submit


def _judge_single(client, question: str, trajectory_text: str, passages_text: str) -> float:
    """Score a single retrieval using LLM judge. Returns 0-1."""
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
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    """Score retrieval quality using LLM judge.

    Extracts full trajectory + submitted passages from each completion.
    Returns 0-1 per completion.
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

        trajectory_text, passages_text, has_submit = _extract_trajectory(completion)

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
