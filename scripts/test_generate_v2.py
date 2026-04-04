#!/usr/bin/env python3
"""Sanity check: generate a few v2 examples (1-hop, 2-hop, 3-hop)."""
import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.data.generate_v2 import (
    generate_training_example,
    SEED_TOPICS_2025,
    DEFAULT_MODEL,
)
from src.env.search_env import SearchEnvironment

def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    env = SearchEnvironment()

    # Test with one of each hop count
    for num_hops in [1, 2, 3]:
        topic = random.choice(SEED_TOPICS_2025)
        print(f"\n{'='*70}")
        print(f"Generating {num_hops}-hop example (topic: {topic})")
        print(f"{'='*70}")

        result = generate_training_example(
            client=client,
            env=env,
            seed_topic=topic,
            model=DEFAULT_MODEL,
            num_hops=num_hops,
            try_expand=(num_hops <= 2),  # try expansion on 1-2 hop
        )

        if result:
            print(f"\n--- RESULT ---")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Sub-answers: {result['sub_answers']}")
            print(f"Ranking: {result['gold_ranking']}")
            print(f"Gold passages ({len(result['gold_passages'])}):")
            for p in result['gold_passages']:
                print(f"  [{p['id']}] {p['source_url'][:60]}")
                print(f"       {p['content'][:100]}...")
            print(f"All snippets: {len(result['all_snippets'])} total")
            if result.get('expanded_from'):
                print(f"Expanded from: {result['expanded_from']}")

            # Save to file
            outfile = f"data/sanity_check_{num_hops}hop.json"
            with open(outfile, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved to {outfile}")
        else:
            print(f"\n--- FAILED to generate {num_hops}-hop example ---")

    print(f"\n{'='*70}")
    print("Sanity check complete!")


if __name__ == "__main__":
    main()
