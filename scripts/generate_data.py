#!/usr/bin/env python3
"""Generate synthetic multi-hop training data using Sonnet + web search.

Usage:
    python scripts/generate_data.py --count 10 --output data/train.jsonl
    python scripts/generate_data.py --count 5 --skip-judge  # faster, no judge pass
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


def main():
    parser = argparse.ArgumentParser(description="Generate multi-hop training questions")
    parser.add_argument("--count", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Anthropic model ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    load_dotenv()
    random.seed(args.seed)

    client = anthropic.Anthropic()
    env = SearchEnvironment()

    output_path = Path(args.output) if args.output else DATA_DIR / "generated.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = []
    attempts = 0
    max_attempts = args.count * 3

    print(f"Generating {args.count} multi-hop questions...")
    print(f"Output: {output_path}")
    print()

    while len(generated) < args.count and attempts < max_attempts:
        attempts += 1
        seed_topic = random.choice(SEED_TOPICS)
        question_type = random.choice(QUESTION_TYPES)

        print(f"[{len(generated)+1}/{args.count}] Topic: {seed_topic} | Type: {question_type}")

        example = generate_training_example(client, env, seed_topic, question_type, model=args.model)

        if example is None:
            print("  ✗ Failed")
            continue

        print(f"  Q: {example['question']}")
        print(f"  A: {example['answer']}")
        j = example.get("judgment", {})
        print(f"  Judge: search={j.get('search_quality')}/5 fetch={j.get('fetch_quality')}/5 "
              f"persistence={j.get('read_persistence')}/5 efficiency={j.get('efficiency')}/5")

        # Format for training: include the trajectory as the conversation
        training_example = {
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Use the search and fetch tools "
                        "to find information. When you have the answer, provide it "
                        "inside <answer></answer> tags."
                    ),
                },
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
            "answer_aliases": [],
            "question_type": example["question_type"],
            "trajectory": example["trajectory"],
            "judgment": example.get("judgment", {}),
            "seed_topic": example["seed_topic"],
        }

        generated.append(training_example)

    # Write output
    with open(output_path, "w") as f:
        for ex in generated:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone. Generated {len(generated)}/{args.count} questions ({attempts} attempts)")
    print(f"Saved to: {output_path}")

    type_counts = {}
    for ex in generated:
        t = ex["question_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Distribution: {type_counts}")


if __name__ == "__main__":
    main()
