"""Format reward — did the trajectory end with a valid submit_answer?

Checks that submitted IDs actually exist in the snippet store (i.e., were
retrieved during the trajectory). This teaches the model to reference real
passages rather than hallucinating IDs.

  0.0 — no submit_answer call, or submitted IDs don't exist in snippet store
  valid/total — partial credit proportional to how many IDs are real
  1.0 — all submitted IDs exist in the snippet store
"""

import re

# Module-level snippet store, set by TiToGRPOTrainer before reward calculation
_snippet_store = None

def set_format_snippet_store(store):
    global _snippet_store
    _snippet_store = store


def _extract_submit_ids(completion: list[dict]) -> list[str] | None:
    """Extract passage_ids from submit_answer call. Returns None if no submit."""
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
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    rewards = []
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
