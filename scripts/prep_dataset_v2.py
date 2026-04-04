#!/usr/bin/env python3
"""Prepare v2 training data for TRL GRPOTrainer.

Converts v2 generated data (with gold_passages, gold_ranking, sub_answers)
into TRL format. The system prompt asks the model to return ranked snippet IDs.

Usage:
    python scripts/prep_dataset_v2.py --input data/train_v2.jsonl --output data/train_trl_v2.jsonl
"""

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a retrieval agent. Use search and read tools to find relevant information. "
    "Each tool result has a snippet ID (like [S1], [R1]). "
    "When done, call submit_ranking with your passage IDs ordered by relevance and source quality."
)


def count_gold_tools(trajectory: list[dict]) -> int:
    """Count tool calls in the trajectory (OpenAI format)."""
    count = 0
    for msg in trajectory:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += len(msg["tool_calls"])
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {args.input}")

    prepped = []
    for ex in examples:
        traj = ex.get("trajectory", [])

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]

        prepped.append({
            "prompt": prompt,
            "answer": ex["answer"],
            "sub_answers": ex.get("sub_answers", []),
            "gold_passages": ex.get("gold_passages", []),
            "gold_ranking": ex.get("gold_ranking", []),
            "gold_tool_count": count_gold_tools(traj),
            "num_hops": ex.get("num_hops", 0),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in prepped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    n_passages = [len(ex["gold_passages"]) for ex in prepped]
    n_tools = [ex["gold_tool_count"] for ex in prepped]
    hop_dist = {}
    for ex in prepped:
        h = ex["num_hops"]
        hop_dist[h] = hop_dist.get(h, 0) + 1

    print(f"Saved {len(prepped)} examples to {args.output}")
    print(f"Hop distribution: {hop_dist}")
    print(f"Avg gold passages: {sum(n_passages)/len(n_passages):.1f}")
    print(f"Avg gold tool calls: {sum(n_tools)/len(n_tools):.1f}")


if __name__ == "__main__":
    main()
