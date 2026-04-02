"""Synthetic multi-hop training data generator.

Pipeline:
  1. Generate question — Sonnet creates a multi-hop question
  2. Search trajectory — Sonnet uses search/fetch tools to answer it,
     we capture the full multi-turn conversation
  3. Judge — separate LLM call evaluates the trajectory quality,
     can trigger re-generation with feedback
  4. Save — question + trajectory + answer + ground truth metadata
"""

import json
import anthropic
from src.env.search_env import SearchEnvironment


# Tool definitions for Sonnet (Anthropic tool_use format)
TOOLS = [
    {
        "name": "search",
        "description": "Search the web. Returns titles and URLs. Use fetch() to read page content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read",
        "description": (
            "Read a web page and find sections matching keywords. "
            "Fetches the page and returns up to 5 excerpts containing the keywords "
            "with surrounding context. Use this to find specific information on a page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read"},
                "keywords": {
                    "type": "string",
                    "description": "Keywords to search for within the page",
                },
            },
            "required": ["url", "keywords"],
        },
    },
]


QUESTION_PROMPT = """\
Generate a multi-hop question that requires chaining EXACTLY 3 facts together.
Topic: {seed_topic}

First, design the 3-hop chain, then write the question.

Format your response EXACTLY like this:
Hop 1: [what to look up] → [answer]
Hop 2: [what to look up using hop 1 result] → [answer]
Hop 3: [what to look up using hop 2 result] → [answer]
Question: [the question whose answer is hop 3's result]

Rules:
- The final answer must be SHORT (a name, number, date, or place)
- Each hop must depend on the previous hop's answer
- Use stable facts (geography, history, science)
- The question must be unambiguous with exactly one correct answer

Examples:
Hop 1: inventor of dynamite → Alfred Nobel
Hop 2: Nobel born in → Sweden
Hop 3: capital of Sweden → Stockholm
Question: What is the capital of the country where the inventor of dynamite was born?

Hop 1: director of Parasite (2019 Best Picture) → Bong Joon-ho
Hop 2: Bong Joon-ho born in → Daegu, South Korea
Hop 3: river flowing through Daegu → Geumho River
Question: What river flows through the birthplace of the director of the film that won Best Picture at the 2020 Academy Awards?\
"""


SEARCH_PROMPT = """\
Answer the following question using the search and read tools.

RULES:
- Your answer MUST come from information you retrieved via search() and read().
- Do NOT answer from your own knowledge. You must search and read to find the answer.
- Use search() to find relevant URLs, then read(url, keywords) to find specific facts on those pages.
- Your final answer must be SHORT — just the name, number, date, or place. No full sentences.

Question: {question}\
"""


JUDGE_PROMPT = """\
You are evaluating a search trajectory for training an AI search agent.
Judge ONLY based on the trajectory below — do not use external knowledge.

Question: {question}
Claimed answer: {answer}

The agent took the following search path:
{trajectory_summary}

Evaluate:
1. Does the answer appear in the retrieved content (snippets or read results)? \
The answer MUST be visible in the trajectory. This is the most important check.
2. Were the search queries effective?
3. Was the path efficient? Not using read() is FINE if snippets were sufficient. \
But using read() many times with bad keywords (getting "No matches found" repeatedly, \
or reading irrelevant pages) is INEFFICIENT and should FAIL.
4. Were all hops of the multi-hop question covered?

IMPORTANT: The verified_answer must be SHORT — just the name, number, date, or place. \
No full sentences.

IMPORTANT: overall_pass should be false if:
- The answer is NOT visible in the retrieved content shown above
- The agent used read() more than 3 times inefficiently

Respond with this JSON:
```json
{{
  "answer_correct": true/false,
  "answer_in_retrieved_content": true/false,
  "verified_answer": "short factual answer only",
  "search_quality": 1-5,
  "retrieval_quality": 1-5,
  "efficiency": 1-5,
  "all_hops_covered": true/false,
  "overall_pass": true/false,
  "feedback": "specific tips for improving the search path if suboptimal"
}}
```\
"""


RETRY_SEARCH_PROMPT = """\
Answer the following question using the search and read tools.

RULES:
- Your answer MUST come from information you retrieved via search() and read().
- Do NOT answer from your own knowledge. You must search and read to find the answer.
- Use search() to find relevant URLs, then read(url, keywords) to find specific facts on those pages.
- Your final answer must be SHORT — just the name, number, date, or place. No full sentences.

Question: {question}

A previous attempt had these issues:
{feedback}

Improve on the previous attempt.\
"""


EXPAND_PROMPT = """\
You are making a multi-hop search question HARDER by adding one more hop that \
requires reading a web page (not just search snippets).

Original question: {question}
Original answer: {answer}

Here are the search results the agent found:
{trajectory_summary}

Look at the URLs and snippets above. Pick one result that likely contains \
interesting details NOT shown in the snippet (e.g., specific numbers, dates, \
names of people, measurements, lesser-known facts).

Rewrite the question to add one more hop that requires reading that page to \
find a specific detail. The new answer must be a SHORT fact (name, number, date).

Format your response EXACTLY like this:
Read URL: [the URL to read for the extra detail]
Read keywords: [2-4 keywords to search for on that page]
New question: [the extended question]
New answer: [short factual answer]\
"""


SEED_TOPICS = [
    "European capital cities and their founding",
    "Major rivers of Asia and the cities they flow through",
    "Countries that changed their name in the 20th century",
    "UNESCO World Heritage sites and their locations",
    "Nobel Prize winners in Physics",
    "Space missions and their launch dates",
    "Inventors and their most famous inventions",
    "Elements on the periodic table and who discovered them",
    "Academy Award winning films and their directors",
    "Famous authors and the cities where they wrote their major works",
    "World record holders in Olympic sports",
    "Major music festivals and where they are held",
    "Constitutional amendments in various countries",
    "Major trade agreements and participating nations",
    "Heads of state during major historical events",
    "Cities that have hosted the Olympic Games",
]

QUESTION_TYPES = ["bridge_entity", "bridge_entity", "comparison", "temporal_entity"]

DEFAULT_MODEL = "claude-sonnet-4-6"


def dispatch_tool(env: SearchEnvironment, name: str, args: dict) -> str:
    if name == "search":
        return env.search(query=args["query"], max_results=args.get("max_results", 5))
    elif name == "read":
        return env.read(url=args["url"], keywords=args["keywords"])
    return f"[Unknown tool: {name}]"


def run_with_tools(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    env: SearchEnvironment,
    model: str = DEFAULT_MODEL,
    max_rounds: int = 15,
    label: str = "",
) -> list[dict]:
    """Run a multi-turn conversation with tools.
    Returns the full message history (the trajectory).
    Each round is one assistant response + tool dispatch. Stops after max_rounds."""
    messages = [{"role": "user", "content": user_message}]
    prefix = f"    [{label}]" if label else "   "

    for round_num in range(1, max_rounds + 1):
        print(f"{prefix} round {round_num}...", flush=True)
        response = client.messages.create(
            model=model, max_tokens=4096, system=system,
            tools=TOOLS, messages=messages,
        )

        # Convert response content to serializable format
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                preview = block.text[:100].replace("\n", " ")
                print(f"{prefix}   text: {preview}", flush=True)
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
                args_short = json.dumps(block.input, ensure_ascii=False)[:100]
                print(f"{prefix}   → {block.name}({args_short})", flush=True)

        messages.append({"role": "assistant", "content": assistant_content})

        # If no tool use, we're done
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        # Dispatch tools and collect results
        tool_results = []
        for tool_use in tool_uses:
            result = dispatch_tool(env, tool_use.name, tool_use.input)
            result_preview = result[:80].replace("\n", " ")
            print(f"{prefix}   ← {result_preview}...", flush=True)
            tool_results.append({
                "type": "tool_result", "tool_use_id": tool_use.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return messages


def extract_final_text(messages: list[dict]) -> str:
    """Get the final text response from a message history."""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block["text"]
            elif isinstance(content, str):
                return content
    return ""


def summarize_trajectory(messages: list[dict]) -> str:
    """Create a human-readable summary of tool calls in a trajectory."""
    lines = []
    for msg in messages:
        if msg["role"] == "assistant":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    args_str = json.dumps(block["input"], ensure_ascii=False)
                    lines.append(f"  → {block['name']}({args_str})")
        elif msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result = block["content"]
                        preview = result[:150] + "..." if len(result) > 150 else result
                        lines.append(f"    ← {preview}")
    return "\n".join(lines)


def extract_json(text: str) -> dict | None:
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# --- Pipeline steps ---

def step_generate_question(
    client: anthropic.Anthropic,
    seed_topic: str,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Step 1: Generate a 3-hop question. Extracts the Question: line from the response."""
    response = client.messages.create(
        model=model, max_tokens=512,
        messages=[{
            "role": "user",
            "content": QUESTION_PROMPT.format(seed_topic=seed_topic),
        }],
    )
    text = response.content[0].text.strip()

    # Extract the "Question:" line
    for line in text.split("\n"):
        if line.strip().startswith("Question:"):
            q = line.split("Question:", 1)[1].strip()
            q = q.strip('"').strip("'")
            return q if q and "?" in q else None
    return None


def step_search_trajectory(
    client: anthropic.Anthropic,
    env: SearchEnvironment,
    question: str,
    feedback: str | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], str]:
    """Step 2: Answer the question using tools.
    Returns (trajectory, final_answer)."""
    env.reset()

    if feedback:
        prompt = RETRY_SEARCH_PROMPT.format(question=question, feedback=feedback)
    else:
        prompt = SEARCH_PROMPT.format(question=question)

    messages = run_with_tools(
        client=client, system="You are a research assistant. Use tools to search the web.",
        user_message=prompt, env=env, model=model, label="search",
    )

    answer = extract_final_text(messages)
    return messages, answer


def step_judge(
    client: anthropic.Anthropic,
    question: str,
    answer: str,
    trajectory: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """Step 3: Judge evaluates the trajectory — no tools, just reads the trajectory."""
    summary = summarize_trajectory(trajectory)
    print("    [judge] evaluating...", flush=True)

    response = client.messages.create(
        model=model, max_tokens=2048,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, answer=answer, trajectory_summary=summary,
            ),
        }],
    )

    judge_text = response.content[0].text.strip()
    return extract_json(judge_text)


def step_expand(
    client: anthropic.Anthropic,
    env: SearchEnvironment,
    question: str,
    answer: str,
    trajectory: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """Step 4 (optional): Expand a snippet-only question by adding a read hop.

    1. LLM proposes an extended question + which URL/keywords to read
    2. We mechanically call read() and append to the trajectory
    Returns expanded example dict or None if expansion fails.
    """
    summary = summarize_trajectory(trajectory)
    print("  Step 4: expanding question...", flush=True)

    response = client.messages.create(
        model=model, max_tokens=512,
        messages=[{
            "role": "user",
            "content": EXPAND_PROMPT.format(
                question=question, answer=answer, trajectory_summary=summary,
            ),
        }],
    )

    text = response.content[0].text.strip()

    # Parse response
    read_url = None
    read_keywords = None
    new_question = None
    new_answer = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Read URL:"):
            read_url = line.split("Read URL:", 1)[1].strip()
        elif line.startswith("Read keywords:"):
            read_keywords = line.split("Read keywords:", 1)[1].strip()
        elif line.startswith("New question:"):
            new_question = line.split("New question:", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("New answer:"):
            new_answer = line.split("New answer:", 1)[1].strip().strip('"').strip("'")

    if not all([read_url, read_keywords, new_question, new_answer]) or "?" not in new_question:
        print("  Step 4: failed to parse expansion", flush=True)
        return None

    print(f"  Step 4 Q: {new_question}", flush=True)
    print(f"  Step 4 A: {new_answer}", flush=True)
    print(f"  Step 4 read: {read_url} [{read_keywords}]", flush=True)

    # Mechanically call read() and append to trajectory
    read_result = env.read(read_url, read_keywords)
    print(f"  Step 4 read result: {read_result[:100]}...", flush=True)

    if read_result == "No matches found.":
        print("  Step 4: read found nothing, skipping expansion", flush=True)
        return None

    # Find where in the trajectory the search returned this URL
    insert_idx = None
    for i, msg in enumerate(trajectory):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if read_url in block.get("content", ""):
                        insert_idx = i + 1  # insert after this tool result
                        break
        if insert_idx is not None:
            break

    if insert_idx is None:
        # URL not found in trajectory — fall back to appending before final answer
        insert_idx = len(trajectory) - 1  # before the last assistant message

    # Build expanded trajectory
    read_call = {
        "role": "assistant",
        "content": [{
            "type": "tool_use", "id": "expand_read",
            "name": "read", "input": {"url": read_url, "keywords": read_keywords},
        }],
    }
    read_response = {
        "role": "user",
        "content": [{
            "type": "tool_result", "tool_use_id": "expand_read",
            "content": read_result,
        }],
    }

    expanded = list(trajectory[:insert_idx])
    expanded.append(read_call)
    expanded.append(read_response)
    expanded.extend(trajectory[insert_idx:-1])  # skip old final answer
    expanded.append({
        "role": "assistant",
        "content": [{"type": "text", "text": new_answer}],
    })

    return {
        "question": new_question,
        "answer": new_answer,
        "trajectory": expanded,
        "expanded_from": question,
    }


def generate_training_example(
    client: anthropic.Anthropic,
    env: SearchEnvironment,
    seed_topic: str,
    question_type: str,
    model: str = DEFAULT_MODEL,
    max_judge_retries: int = 2,
) -> dict | None:
    """Full pipeline: generate question → search → judge → (retry with feedback) → save.

    The judge can reject a trajectory up to max_judge_retries times.
    Each rejection feeds back tips to improve the next search attempt.
    """

    # Step 1: Generate question
    print("  Step 1: generating question...", flush=True)
    question = step_generate_question(client, seed_topic, model=model)
    if not question:
        print("  Step 1: failed to generate question", flush=True)
        return None
    print(f"  Step 1: {question}", flush=True)

    feedback = None
    trajectory = None
    judgment = None

    for attempt in range(1 + max_judge_retries):
        # Step 2: Search trajectory
        retry_note = f" (retry {attempt}, feedback: {feedback[:60]}...)" if feedback else ""
        print(f"  Step 2: search trajectory{retry_note}", flush=True)
        trajectory, answer = step_search_trajectory(client, env, question, feedback=feedback, model=model)
        if not answer:
            print("  Step 2: no answer produced", flush=True)
            return None
        print(f"  Step 2 answer: {answer[:100]}", flush=True)

        # Step 3: Judge
        print("  Step 3: judging...", flush=True)
        judgment = step_judge(client, question, answer, trajectory, model=model)
        if judgment is None:
            print("  Step 3: judge failed to parse", flush=True)
            return None

        passed = judgment.get("overall_pass", False) and judgment.get("answer_correct", False)
        print(f"  Step 3: pass={passed}", flush=True)

        if passed:
            break

        # Otherwise, feed back for next attempt
        feedback = judgment.get("feedback", "Ensure the answer is grounded in retrieved content.")
        print(f"  Step 3: retrying — {feedback[:80]}", flush=True)

    if judgment is None or not judgment.get("answer_correct", False):
        return None

    # Check if trajectory used any reads
    has_reads = any(
        block.get("name") == "read"
        for msg in trajectory if msg["role"] == "assistant"
        for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict)
    )

    # If no reads were used, try expanding the question to require one
    if not has_reads:
        expanded = step_expand(
            client, env, question, judgment.get("verified_answer", answer),
            trajectory, model=model,
        )
        if expanded:
            return {
                "question": expanded["question"],
                "question_type": question_type,
                "answer": expanded["answer"],
                "trajectory": expanded["trajectory"],
                "judgment": judgment,
                "seed_topic": seed_topic,
                "expanded_from": expanded["expanded_from"],
            }

    return {
        "question": question,
        "question_type": question_type,
        "answer": judgment.get("verified_answer", answer),
        "trajectory": trajectory,
        "judgment": judgment,
        "seed_topic": seed_topic,
    }
