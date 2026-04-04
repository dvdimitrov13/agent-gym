#!/usr/bin/env python3
"""Generate v2 training/eval data — 2025 events, snippet-ID ranking."""
import argparse
import json
import os
import random
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.data.generate_v2 import (
    generate_training_example,
    SEED_TOPICS_2025,
    DEFAULT_MODEL,
)
from src.env.search_env import SearchEnvironment


def parse_hop_distribution(s: str) -> dict[int, int]:
    """Parse '1:8,2:10,3:7' into {1: 8, 2: 10, 3: 7}."""
    result = {}
    for pair in s.split(","):
        hops, count = pair.split(":")
        result[int(hops)] = int(count)
    return result


def count_existing(path: str) -> dict[int, int]:
    """Count existing examples by hop count."""
    counts = {1: 0, 2: 0, 3: 0}
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                h = d.get("num_hops", 0)
                if h in counts:
                    counts[h] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--hops", type=str, default="1:8,2:10,3:7",
                        help="Distribution like 1:8,2:10,3:7")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    env = SearchEnvironment()
    target = parse_hop_distribution(args.hops)
    existing = count_existing(args.output)

    total_target = sum(target.values())
    total_existing = sum(existing.values())
    print(f"Target: {target} = {total_target} total")
    print(f"Existing: {existing} = {total_existing} total")

    # Build work queue
    work = []
    for hops, count in sorted(target.items()):
        remaining = count - existing.get(hops, 0)
        for _ in range(max(0, remaining)):
            work.append(hops)
    random.shuffle(work)

    if not work:
        print("All examples already generated!")
        return

    max_attempts_per_slot = 5  # try up to 5 different questions before giving up on a slot

    print(f"Generating {len(work)} more examples (max {max_attempts_per_slot} attempts each)...")
    generated = 0
    total_failed = 0
    slot_num = 0

    while work:
        num_hops = work[0]
        slot_num += 1
        success = False

        for attempt in range(1, max_attempts_per_slot + 1):
            topic = random.choice(SEED_TOPICS_2025)
            attempt_label = f" (attempt {attempt})" if attempt > 1 else ""
            print(f"\n[{slot_num}/{slot_num + len(work) - 1}] {num_hops}-hop{attempt_label} (topic: {topic})")

            t0 = time.time()
            try:
                result = generate_training_example(
                    client=client,
                    env=env,
                    seed_topic=topic,
                    model=args.model,
                    num_hops=num_hops,
                    try_expand=(num_hops <= 2),
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                total_failed += 1
                continue

            elapsed = time.time() - t0

            if result:
                with open(args.output, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                generated += 1
                n_passages = len(result.get("gold_passages", []))
                n_snippets = len(result.get("all_snippets", {}))
                print(f"  OK ({elapsed:.0f}s) — {result['answer']}, "
                      f"{n_passages} gold passages, {n_snippets} total snippets")
                success = True
                break
            else:
                total_failed += 1
                print(f"  FAILED ({elapsed:.0f}s) — will retry with different question")

        work.pop(0)
        if not success:
            print(f"  GAVE UP on {num_hops}-hop slot after {max_attempts_per_slot} attempts")

    print(f"\nDone: {generated} generated, {total_failed} failed attempts")
    final = count_existing(args.output)
    print(f"Final counts: {final} = {sum(final.values())} total")


if __name__ == "__main__":
    main()
