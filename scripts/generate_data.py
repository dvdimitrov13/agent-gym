#!/usr/bin/env python3
"""Generate synthetic training/eval data using Sonnet + web search.

Usage:
    # 10 eval questions: 3 single-hop, 4 two-hop, 3 three-hop
    python scripts/generate_data.py --hops 1:3,2:4,3:3 --output data/eval.jsonl

    # Default: 10 three-hop questions
    python scripts/generate_data.py --count 10 --output data/train.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.config import DATA_DIR
from src.env.search_env import SearchEnvironment
from src.data.generate import (
    SEED_TOPICS,
    QUESTION_TYPES,
    generate_training_example,
)


def parse_hops(hops_str: str) -> list[tuple[int, int]]:
    """Parse '1:3,2:4,3:3' into [(1,3), (2,4), (3,3)]."""
    pairs = []
    for part in hops_str.split(","):
        hops, count = part.strip().split(":")
        pairs.append((int(hops), int(count)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate training/eval questions")
    parser.add_argument("--count", type=int, default=None, help="Number of questions (all same hop count)")
    parser.add_argument("--hops", type=str, default=None,
                        help="Hop distribution, e.g. '1:3,2:4,3:3' (hop_count:quantity)")
    parser.add_argument("--num-hops", type=int, default=3, help="Hop count when using --count (default 3)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Anthropic model ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Build work plan: list of (num_hops, quantity)
    if args.hops:
        work = parse_hops(args.hops)
    elif args.count:
        work = [(args.num_hops, args.count)]
    else:
        parser.error("Provide either --hops or --count")
        return

    total = sum(qty for _, qty in work)

    load_dotenv()
    random.seed(args.seed)

    client = anthropic.Anthropic()
    env = SearchEnvironment()

    output_path = Path(args.output) if args.output else DATA_DIR / "generated.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing examples and count by hop
    generated = []
    existing_hops = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    generated.append(ex)
                    h = ex.get("num_hops", 3)
                    existing_hops[h] = existing_hops.get(h, 0) + 1
        if generated:
            print(f"Resuming: found {len(generated)} existing examples: {existing_hops}")

    global_idx = len(generated)

    for num_hops, qty in work:
        already = existing_hops.get(num_hops, 0)
        if already >= qty:
            print(f"\nSkipping {num_hops}-hop: already have {already}/{qty}")
            continue
        remaining = qty - already
        print(f"\n{'='*60}")
        print(f"Generating {remaining} x {num_hops}-hop questions" +
              (f" ({already} already done)" if already else ""))
        print(f"{'='*60}\n")

        count = 0
        attempts = 0
        max_attempts = remaining * 3

        while count < remaining and attempts < max_attempts:
            attempts += 1
            global_idx += 1
            seed_topic = random.choice(SEED_TOPICS)
            question_type = random.choice(QUESTION_TYPES)

            print(f"[{global_idx}/{total}] {num_hops}-hop | Topic: {seed_topic} | Type: {question_type}")

            example = generate_training_example(
                client, env, seed_topic, question_type,
                model=args.model, num_hops=num_hops,
            )

            if example is None:
                print("  ✗ Failed\n")
                continue

            print(f"  Q: {example['question']}")
            print(f"  A: {example['answer']}")
            j = example.get("judgment", {})
            print(f"  Judge: search={j.get('search_quality')}/5 "
                  f"retrieval={j.get('retrieval_quality')}/5 "
                  f"efficiency={j.get('efficiency')}/5")
            if example.get("expanded_from"):
                print(f"  (expanded from: {example['expanded_from'][:80]})")
            print()

            training_example = {
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Use the search and read tools "
                            "to find information. When you have the answer, provide it "
                            "inside <answer></answer> tags."
                        ),
                    },
                    {"role": "user", "content": example["question"]},
                ],
                "answer": example["answer"],
                "answer_aliases": [],
                "num_hops": num_hops,
                "question_type": example["question_type"],
                "trajectory": example["trajectory"],
                "judgment": example.get("judgment", {}),
                "seed_topic": example["seed_topic"],
                "expanded_from": example.get("expanded_from"),
            }

            generated.append(training_example)
            count += 1

            # Save incrementally — append this example immediately
            with open(output_path, "a") as f:
                f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            print(f"  ✓ Saved ({len(generated)}/{total} total)\n")

        if count < remaining:
            print(f"  ⚠ Only generated {count}/{remaining} {num_hops}-hop questions after {attempts} attempts")

    print(f"\nDone. Generated {len(generated)}/{total} questions")
    print(f"Saved to: {output_path}")

    hop_counts = {}
    for ex in generated:
        h = ex["num_hops"]
        hop_counts[h] = hop_counts.get(h, 0) + 1
    print(f"Distribution: {hop_counts}")


if __name__ == "__main__":
    main()
