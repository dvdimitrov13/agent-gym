#!/usr/bin/env python3
"""Regenerate flagged examples with research-grounded questions + inline DQ check.

Reads the clean dataset and flagged list, regenerates replacements
with the research step, DQ checks each one, retries on failure.

Usage:
    python scripts/regen_flagged.py \
        --clean data/train_v2_clean.jsonl \
        --flagged data/train_v2_clean_flagged.json \
        --output data/train_v2_final.jsonl
"""
import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

from src.data.generate_v2 import (
    generate_training_example, SEED_TOPICS_2025, DEFAULT_MODEL,
)
from src.env.search_env import SearchEnvironment

# Inline DQ judge (same as dq_check.py but synchronous for per-example use)
DQ_PROMPT = """\
You are a data quality judge. Check this training example for issues:
1. FALSE PREMISE: Question assumes something that didn't happen
2. UNGROUNDED: Answer not supported by gold passages
3. CORRECTION: Answer corrects the question instead of answering it
4. HALLUCINATED: Event doesn't appear to have happened

Question: {question}
Answer: {answer}
Gold passages (first 3):
{passages}

Respond with JSON: {{"pass": true/false, "issue": "none" or issue type, "explanation": "brief"}}"""


def dq_check_single(client: OpenAI, example: dict, model: str = DEFAULT_MODEL) -> dict:
    """Quick DQ check on a single example."""
    passages = example.get("gold_passages", [])
    passages_text = "\n".join(
        f"  [{p.get('id', '?')}] {p.get('content', '')[:150]}"
        for p in passages[:3]
    ) or "(none)"

    response = client.chat.completions.create(
        model=model, max_completion_tokens=200,
        messages=[{"role": "user", "content": DQ_PROMPT.format(
            question=example["question"],
            answer=example["answer"],
            passages=passages_text,
        )}],
    )
    text = response.choices[0].message.content.strip()
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except:
        return {"pass": True, "issue": "parse_error"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="data/train_v2_clean.jsonl")
    parser.add_argument("--flagged", default="data/train_v2_clean_flagged.json")
    parser.add_argument("--output", default="data/train_v2_final.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    env = SearchEnvironment()

    # Load clean examples
    with open(args.clean) as f:
        clean = [json.loads(line) for line in f if line.strip()]
    print(f"Clean examples: {len(clean)}")

    # Load flagged
    with open(args.flagged) as f:
        flagged = json.load(f)
    hop_counts = Counter(b["num_hops"] for b in flagged)
    print(f"Flagged: {len(flagged)} — by hops: {dict(hop_counts)}")

    # Build regeneration queue
    queue = []
    for h, count in sorted(hop_counts.items()):
        for _ in range(count):
            queue.append(h)
    random.shuffle(queue)

    # Regenerate with DQ check
    regenerated = []
    total_attempts = 0
    total_dq_fails = 0

    for i, num_hops in enumerate(queue):
        topic = random.choice(SEED_TOPICS_2025)
        print(f"\n[{i+1}/{len(queue)}] Regenerating {num_hops}-hop (topic: {topic})")

        for attempt in range(1, args.max_attempts + 1):
            total_attempts += 1
            t0 = time.time()

            try:
                result = generate_training_example(
                    client=client, env=env, seed_topic=topic,
                    model=args.model, num_hops=num_hops, try_expand=(num_hops <= 2),
                )
            except Exception as e:
                print(f"  Attempt {attempt}: ERROR {e}")
                topic = random.choice(SEED_TOPICS_2025)
                continue

            if not result:
                print(f"  Attempt {attempt}: generation failed ({time.time()-t0:.0f}s)")
                topic = random.choice(SEED_TOPICS_2025)
                continue

            # Inline DQ check
            dq = dq_check_single(client, result, model=args.model)
            if dq.get("pass", True):
                regenerated.append(result)
                print(f"  Attempt {attempt}: OK — {result['answer'][:50]} ({time.time()-t0:.0f}s)")
                break
            else:
                total_dq_fails += 1
                print(f"  Attempt {attempt}: DQ FAIL — {dq.get('issue')}: {dq.get('explanation', '')[:60]} ({time.time()-t0:.0f}s)")
                topic = random.choice(SEED_TOPICS_2025)
        else:
            print(f"  GAVE UP after {args.max_attempts} attempts")

    # Combine clean + regenerated
    final = clean + regenerated
    # Sort by hops for curriculum
    final.sort(key=lambda x: x.get("num_hops", 0))

    with open(args.output, "w") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    hop_dist = Counter(ex.get("num_hops", 0) for ex in final)
    print(f"\nDone: {len(regenerated)} regenerated ({total_attempts} attempts, {total_dq_fails} DQ fails)")
    print(f"Final dataset: {len(final)} examples — hops: {dict(hop_dist)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
