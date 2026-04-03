#!/usr/bin/env python3
"""Prepare training data for TRL GRPOTrainer.

Reads the generated JSONL and adds gold_urls + gold_tool_count columns
extracted from reference trajectories.

Usage:
    python scripts/prep_dataset.py --input data/train.jsonl --output data/train_trl.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def extract_gold_urls(trajectory: list[dict]) -> list[str]:
    """Extract main result URLs from search results in the trajectory."""
    urls = set()
    for msg in trajectory:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = block.get("content", "")
            for match in re.finditer(r"^\s+(https?://\S+)", text, re.MULTILINE):
                urls.add(match.group(1))
    # Also add URLs from read() calls
    for msg in trajectory:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if (isinstance(block, dict) and block.get("type") == "tool_use"
                    and block.get("name") == "read"):
                url = block.get("input", {}).get("url", "")
                if url:
                    urls.add(url)
    return list(urls)


def count_gold_tools(trajectory: list[dict]) -> int:
    """Count tool_use blocks in the trajectory."""
    count = 0
    for msg in trajectory:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for TRL training")
    parser.add_argument("--input", type=str, default="data/train.jsonl")
    parser.add_argument("--output", type=str, default="data/train_trl.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    examples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {input_path}")

    SYSTEM_PROMPT = (
        "You are a research assistant. Use search and read tools to find information. "
        "Think briefly, then act. Be concise — do not repeat retrieved content. "
        "When you have the answer, state it directly."
    )

    prepped = []
    for ex in examples:
        traj = ex.get("trajectory", [])
        # Override the system prompt to encourage conciseness
        prompt = list(ex["prompt"])
        prompt[0] = {"role": "system", "content": SYSTEM_PROMPT}
        prepped.append({
            "prompt": prompt,
            "answer": ex["answer"],
            "answer_aliases": ex.get("answer_aliases", []),
            "gold_urls": extract_gold_urls(traj),
            "gold_tool_count": count_gold_tools(traj),
            "num_hops": ex.get("num_hops", 0),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in prepped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    tool_counts = [ex["gold_tool_count"] for ex in prepped]
    url_counts = [len(ex["gold_urls"]) for ex in prepped]
    print(f"Saved {len(prepped)} examples to {output_path}")
    print(f"Avg gold tool calls: {sum(tool_counts)/len(tool_counts):.1f}")
    print(f"Avg gold URLs: {sum(url_counts)/len(url_counts):.1f}")


if __name__ == "__main__":
    main()
