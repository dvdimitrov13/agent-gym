#!/usr/bin/env python3
"""Test reward functions on eval data and M3 baseline results.

Usage:
    python scripts/test_rewards.py
"""

import json
import re
from pathlib import Path

from src.rewards import retrieval_reward, efficiency_reward, thinking_reward


def extract_gold_urls(trajectory: list[dict]) -> list[str]:
    """Extract main result URLs from a reference trajectory's search results."""
    urls = []
    for msg in trajectory:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = block.get("content", "")
            for match in re.finditer(r"^\s+(https?://\S+)", text, re.MULTILINE):
                urls.append(match.group(1))
    return list(set(urls))


def count_gold_tools(trajectory: list[dict]) -> int:
    """Count tool calls in the reference trajectory."""
    count = 0
    for msg in trajectory:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                count += 1
    return count


def print_scores(label, examples, completions, answers, gold_urls_list, gold_tool_counts, extra_info=None):
    r_retrieval = retrieval_reward(completions, answers, gold_urls=gold_urls_list)
    r_efficiency = efficiency_reward(completions, gold_tool_count=gold_tool_counts)
    r_thinking = thinking_reward(completions)

    print(f"\n{'='*60}")
    print(label)
    print("=" * 60)

    for i, ex in enumerate(examples):
        q = ex["prompt"][1]["content"][:60]
        info = f"  {extra_info[i]}" if extra_info else ""
        print(f"  [{i+1}] {q}")
        print(f"       retrieval={r_retrieval[i]:.2f}  efficiency={r_efficiency[i]:.1f}  "
              f"thinking={r_thinking[i]:.2f}{info}")

    n = len(completions)
    print(f"\n  Avg: retrieval={sum(r_retrieval)/n:.2f}  "
          f"efficiency={sum(r_efficiency)/n:.2f}  "
          f"thinking={sum(r_thinking)/n:.2f}")


def main():
    eval_path = Path("data/eval.jsonl")
    if not eval_path.exists():
        print("No eval data found")
        return

    examples = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} eval examples")

    answers = [ex["answer"] for ex in examples]
    gold_urls_list = [extract_gold_urls(ex["trajectory"]) for ex in examples]
    gold_tool_counts = [count_gold_tools(ex["trajectory"]) for ex in examples]

    # Score reference trajectories
    trajectories = [ex["trajectory"] for ex in examples]
    print_scores("REFERENCE trajectories (Sonnet)", examples, trajectories,
                 answers, gold_urls_list, gold_tool_counts)

    # Score 0.6B baseline
    m3_path = Path("results/m3_baseline_with_tools.json")
    if m3_path.exists():
        with open(m3_path) as f:
            m3_data = json.load(f)

        model_completions = []
        extra = []
        for r in m3_data["results"]:
            predicted = r.get("predicted", "") or ""
            msgs = []
            for tc in r.get("tool_calls", []):
                msgs.append({
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "x",
                                 "name": tc["name"], "input": tc["arguments"]}],
                })
                msgs.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "x", "content": ""}],
                })
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": predicted}]})
            model_completions.append(msgs)
            extra.append(f"pred={predicted[:40]}..." if len(predicted) > 40 else f"pred={predicted}")

        print_scores("0.6B MODEL (M3 baseline)", examples, model_completions,
                     answers, gold_urls_list, gold_tool_counts, extra)

    # Score 14B baseline
    m3_14b_path = Path("results/m3_qwen3_14b.json")
    if m3_14b_path.exists():
        with open(m3_14b_path) as f:
            m3_14b = json.load(f)

        model_completions = []
        extra = []
        for r in m3_14b["results"]:
            predicted = r.get("predicted", "") or ""
            msgs = []
            for tc in r.get("tool_calls", []):
                msgs.append({
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "x",
                                 "name": tc["name"], "input": tc["arguments"]}],
                })
                msgs.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "x", "content": ""}],
                })
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": predicted}]})
            model_completions.append(msgs)
            extra.append(f"pred={predicted[:40]}..." if len(predicted) > 40 else f"pred={predicted}")

        print_scores("14B MODEL (M3 baseline)", examples, model_completions,
                     answers, gold_urls_list, gold_tool_counts, extra)


if __name__ == "__main__":
    main()
