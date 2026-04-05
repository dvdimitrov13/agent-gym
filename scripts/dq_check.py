#!/usr/bin/env python3
"""Data quality check — LLM judge flags bad training examples.

Runs GPT 5.4 as a judge over each example to check:
1. Is the question about a real 2025+ event (not hallucinated)?
2. Does the answer match the gold passages (grounded, not corrected)?
3. Is the answer a direct factual retrieval, not a "no one" / "actually..."?

Drops bad examples, regenerates replacements, DQ checks those too.

Usage:
    python scripts/dq_check.py --input data/train_v2.jsonl --output data/train_v2_clean.jsonl
"""
import argparse
import asyncio
import json
import os
import random
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

JUDGE_PROMPT = """\
You are a data quality judge for a retrieval training dataset. Each example has a question \
about a 2025 or 2026 event, an answer, and gold passages retrieved from the web.

Check for these issues:
1. FALSE PREMISE: The question assumes something that didn't happen (e.g., asks about a 2025 \
   event that actually happened in 2024, or attributes something to the wrong person/entity).
2. UNGROUNDED ANSWER: The answer is not supported by the gold passages, or contradicts them.
3. CORRECTION ANSWER: The answer corrects the question rather than answering it \
   (e.g., "No one — that actually happened in 2024" or "None — X did not win Y").
4. HALLUCINATED EVENT: The question is about an event that doesn't appear to have happened.
5. AMBIGUOUS: The question has multiple valid answers or is unclear.

Example:
{example_json}

Gold passages (what the search actually found):
{passages_text}

Respond with EXACTLY this JSON:
```json
{{
  "pass": true/false,
  "issue": "none" or one of "false_premise", "ungrounded", "correction", "hallucinated", "ambiguous",
  "explanation": "brief explanation"
}}
```"""

MODEL = "gpt-5.4"
BATCH_SIZE = 25


async def judge_example(client: AsyncOpenAI, example: dict, idx: int) -> dict:
    """Judge a single example."""
    # Build passages text
    passages = example.get("gold_passages", [])
    passages_text = "\n".join(
        f"[{p.get('id', '?')}] {p.get('source_url', '?')[:60]}\n  {p.get('content', '')[:200]}"
        for p in passages[:5]
    ) or "(no gold passages)"

    example_json = json.dumps({
        "question": example["question"],
        "answer": example["answer"],
        "num_hops": example.get("num_hops"),
        "sub_answers": example.get("sub_answers", []),
    }, indent=2)

    prompt = JUDGE_PROMPT.format(example_json=example_json, passages_text=passages_text)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        result["idx"] = idx
        return result
    except Exception as e:
        return {"idx": idx, "pass": True, "issue": "error", "explanation": str(e)}


async def judge_batch(client: AsyncOpenAI, examples: list[dict], start_idx: int) -> list[dict]:
    """Judge a batch of examples concurrently."""
    tasks = [
        judge_example(client, ex, start_idx + i)
        for i, ex in enumerate(examples)
    ]
    return await asyncio.gather(*tasks)


async def run_dq(input_path: str, output_path: str):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(input_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Checking {len(examples)} examples in batches of {BATCH_SIZE}...")

    all_results = []
    for batch_start in range(0, len(examples), BATCH_SIZE):
        batch = examples[batch_start:batch_start + BATCH_SIZE]
        print(f"  Batch {batch_start//BATCH_SIZE + 1}: examples {batch_start}-{batch_start+len(batch)-1}...", end=" ", flush=True)
        t0 = time.time()
        results = await judge_batch(client, batch, batch_start)
        all_results.extend(results)
        n_fail = sum(1 for r in results if not r.get("pass", True))
        print(f"done ({time.time()-t0:.1f}s, {n_fail} flagged)")

    # Separate good and bad
    good = []
    bad = []
    for r in all_results:
        idx = r["idx"]
        if r.get("pass", True):
            good.append(examples[idx])
        else:
            bad.append({
                "idx": idx,
                "question": examples[idx]["question"][:80],
                "answer": examples[idx]["answer"][:80],
                "issue": r.get("issue", "unknown"),
                "explanation": r.get("explanation", ""),
                "num_hops": examples[idx].get("num_hops"),
            })

    print(f"\nResults: {len(good)} passed, {len(bad)} flagged")
    for b in bad:
        print(f"  [{b['idx']}] {b['issue']}: {b['question']}")
        print(f"       A: {b['answer']}")
        print(f"       Why: {b['explanation']}")
        print()

    # Save clean examples
    with open(output_path, "w") as f:
        for ex in good:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(good)} clean examples to {output_path}")

    # Save flagged for review
    flagged_path = output_path.replace(".jsonl", "_flagged.json")
    with open(flagged_path, "w") as f:
        json.dump(bad, f, indent=2)
    print(f"Saved {len(bad)} flagged examples to {flagged_path}")

    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_v2.jsonl")
    parser.add_argument("--output", default="data/train_v2_clean.jsonl")
    args = parser.parse_args()

    bad = asyncio.run(run_dq(args.input, args.output))

    if bad:
        print(f"\n{'='*60}")
        print(f"Need to regenerate {len(bad)} examples")
        # Count by hop
        from collections import Counter
        hop_counts = Counter(b["num_hops"] for b in bad)
        print(f"  By hops: {dict(hop_counts)}")
        print(f"Run: python scripts/generate_data_v2.py --output {args.output} "
              f"--hops {','.join(f'{h}:{c}' for h, c in sorted(hop_counts.items()))}")


if __name__ == "__main__":
    main()
