"""Format reward — validates submit_answer passage IDs against the snippet store.

Used as a multiplicative scaler on the judge reward. Checks that submitted
IDs actually exist in the snippet store (i.e., were retrieved during the
trajectory). This prevents the model from hallucinating passage IDs.

Scoring:
  - 0.0 — no submit_answer call, or submitted IDs don't exist
  - valid/total — partial credit proportional to valid IDs
  - 1.0 — all submitted IDs reference real retrieved passages
"""

from __future__ import annotations

import json
import re
from typing import Any

# Module-level snippet store, set by TiToGRPOTrainer before reward calculation.
_snippet_store: dict[int, dict[str, str]] | None = None


def set_format_snippet_store(store: dict[int, dict[str, str]] | None) -> None:
    """Set the snippet store for format reward validation.

    Args:
        store: Mapping of completion index to ``{snippet_id: content}``.
    """
    global _snippet_store
    _snippet_store = store


def _extract_submit_ids(completion: list[dict[str, Any]]) -> list[str] | None:
    """Extract passage IDs from a submit_answer tool call in the completion.

    Handles three formats: structured tool_calls, ``<tool_call>`` JSON blocks,
    and raw text ``submit_answer([...])`` calls.

    Args:
        completion: List of message dicts from a single rollout.

    Returns:
        List of passage ID strings, or None if no submit_answer was called.
    """
    for msg in completion:
        if msg.get("role") != "assistant":
            continue
        # Check structured tool_calls
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            if func.get("name") == "submit_answer":
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        import json
                        args = json.loads(args)
                    except:
                        return []
                return args.get("passage_ids", [])
        # Check raw text content for both formats
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        # Format 1: <tool_call>{"name":"submit_answer","arguments":{"passage_ids":[...]}}</tool_call>
        if "<tool_call>" in content and "submit_answer" in content:
            import json as _json
            for tc_match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL):
                try:
                    tc_data = _json.loads(tc_match.group(1))
                    if tc_data.get("name") == "submit_answer":
                        args = tc_data.get("arguments", {})
                        if isinstance(args, str):
                            args = _json.loads(args)
                        if isinstance(args, dict):
                            return [str(x) for x in args.get("passage_ids", [])]
                        elif isinstance(args, list):
                            return [str(x) for x in args]
                except (_json.JSONDecodeError, TypeError):
                    pass

        # Format 2: submit_answer([S1, S2, ...])
        if "submit_answer" in content:
            m = re.search(r"submit_answer\(\s*\[(.+?)\]\s*\)", content, re.DOTALL)
            if m:
                raw = m.group(1)
                ids = [s.strip().strip("'\"") for s in raw.split(",")]
                return [str(x) for x in ids]
            return []
    return None


def format_reward(
    completions: list[list[dict[str, Any]]],
    **kwargs: Any,
) -> list[float]:
    """Score format validity for each completion.

    Args:
        completions: List of rollout message lists.

    Returns:
        List of scores in [0, 1], one per completion.
    """
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        ids = _extract_submit_ids(completion)
        if ids is None:
            rewards.append(0.0)
            continue

        if not ids:
            rewards.append(0.0)
            continue

        # Check if IDs exist in the snippet store for this completion
        available = set()
        if _snippet_store and idx in _snippet_store:
            available = set(_snippet_store[idx].keys())

        if not available:
            # No snippet store available — fall back to submit-only signal
            rewards.append(1.0)
            continue

        valid = sum(1 for pid in ids if pid in available)
        rewards.append(valid / len(ids))

    return rewards
