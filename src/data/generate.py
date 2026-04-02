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
        "name": "fetch",
        "description": (
            "Read a specific page of content from a URL, like reading a book. "
            "Content is split into pages of ~500 words. Page 1 is the beginning, "
            "page 2 is the next section, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read"},
                "page": {
                    "type": "integer",
                    "description": "Which page to read (1-indexed, default 1)",
                    "default": 1,
                },
            },
            "required": ["url"],
        },
    },
]


QUESTION_PROMPT = """\
Generate a multi-hop question that requires chaining at least 3 facts together.
Topic: {seed_topic}
Type: {question_type}

The question must:
- Require at least 3 separate lookups (3 hops) to answer
- Have a SHORT factual answer (a name, number, date, or place — under 10 words)
- Be unambiguous — exactly one correct answer
- Use stable facts (geography, history, science) not things that change

3-hop examples:
- "What language is spoken in the birthplace of the scientist who discovered Neptunium?"
  Hop 1: who discovered Neptunium → Edwin McMillan
  Hop 2: where was McMillan born → Redondo Beach, California
  Hop 3: language spoken there → English

- "In what year did the Olympic Games take place in the country where the architect of the Sydney Opera House was born?"
  Hop 1: architect of Sydney Opera House → Jorn Utzon
  Hop 2: where was Utzon born → Denmark
  Hop 3: Olympics in Denmark → 1920 (not hosted, so this is a bad question — avoid these)

- "What is the capital of the country where the inventor of dynamite was born?"
  Hop 1: inventor of dynamite → Alfred Nobel
  Hop 2: Nobel born in → Sweden
  Hop 3: capital of Sweden → Stockholm

Respond with ONLY the question, nothing else.\
"""


SEARCH_PROMPT = """\
Answer the following question using the search and fetch tools.

RULES:
- Your answer MUST come from information you retrieved (search snippets or fetched pages).
- Do NOT answer from your own knowledge. If you already know the answer, still search to find a source.
- Use fetch() to read pages when snippets are not sufficient.
- Your final answer must be SHORT — just the name, number, date, or place. No full sentences.

Question: {question}\
"""


JUDGE_PROMPT = """\
You are evaluating a search trajectory for training an AI search agent.

Question: {question}
Claimed answer: {answer}

The agent took the following search path:
{trajectory_summary}

Evaluate:
1. Is the answer correct? (verify with your own search)
2. Does the answer appear in the retrieved content? Look at the search snippets and \
fetched page text — the answer MUST be grounded in what was actually retrieved, not \
the agent's internal knowledge. This is the most important check.
3. Were the search queries effective? (good keywords, not too vague)
4. Did the agent read pages when snippets were insufficient?
5. Was the path efficient? (no unnecessary searches/fetches)
6. Were all hops of the multi-hop question covered?

IMPORTANT: The verified_answer must be SHORT — just the name, number, date, or place. \
No full sentences.

IMPORTANT: overall_pass must be false if the answer is NOT visible in any of the \
retrieved content (snippets or fetched pages). We need answers grounded in retrieval.

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
Answer the following question using the search and fetch tools.

RULES:
- Your answer MUST come from information you retrieved (search snippets or fetched pages).
- Do NOT answer from your own knowledge. If you already know the answer, still search to find a source.
- Use fetch() to read pages when snippets are not sufficient.
- Your final answer must be SHORT — just the name, number, date, or place. No full sentences.

Question: {question}

A previous attempt had these issues:
{feedback}

Improve on the previous attempt.\
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
    elif name == "fetch":
        return env.fetch(url=args["url"], page=args.get("page", 1))
    return f"[Unknown tool: {name}]"


def run_with_tools(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    env: SearchEnvironment,
    model: str = DEFAULT_MODEL,
    max_rounds: int = 15,
) -> list[dict]:
    """Run a multi-turn conversation with tools.
    Returns the full message history (the trajectory).
    Each round is one assistant response + tool dispatch. Stops after max_rounds."""
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_rounds):
        response = client.messages.create(
            model=model, max_tokens=4096, system=system,
            tools=TOOLS, messages=messages,
        )

        # Convert response content to serializable format
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        # If no tool use, we're done
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        # Dispatch tools and collect results
        tool_results = []
        for tool_use in tool_uses:
            result = dispatch_tool(env, tool_use.name, tool_use.input)
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
    question_type: str,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Step 1: Generate a multi-hop question (no tools needed)."""
    response = client.messages.create(
        model=model, max_tokens=256,
        messages=[{
            "role": "user",
            "content": QUESTION_PROMPT.format(
                seed_topic=seed_topic, question_type=question_type,
            ),
        }],
    )
    text = response.content[0].text.strip()
    # Should be just the question — strip quotes if present
    text = text.strip('"').strip("'")
    return text if text and "?" in text else None


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
        user_message=prompt, env=env, model=model,
    )

    answer = extract_final_text(messages)
    return messages, answer


def step_judge(
    client: anthropic.Anthropic,
    env: SearchEnvironment,
    question: str,
    answer: str,
    trajectory: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """Step 3: Judge evaluates the trajectory quality."""
    env.reset()
    summary = summarize_trajectory(trajectory)

    judge_messages = run_with_tools(
        client=client,
        system="You are a search trajectory judge. Use tools to verify answers.",
        user_message=JUDGE_PROMPT.format(
            question=question, answer=answer, trajectory_summary=summary,
        ),
        env=env, model=model,
    )

    judge_text = extract_final_text(judge_messages)
    return extract_json(judge_text)


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
    question = step_generate_question(client, seed_topic, question_type, model=model)
    if not question:
        return None

    feedback = None
    trajectory = None
    judgment = None

    for attempt in range(1 + max_judge_retries):
        # Step 2: Search trajectory
        trajectory, answer = step_search_trajectory(client, env, question, feedback=feedback, model=model)
        if not answer:
            return None

        # Step 3: Judge
        judgment = step_judge(client, env, question, answer, trajectory, model=model)
        if judgment is None:
            return None

        # If passed, we're done
        if judgment.get("overall_pass", False) and judgment.get("answer_correct", False):
            break

        # Otherwise, feed back for next attempt
        feedback = judgment.get("feedback", "Ensure the answer is grounded in retrieved content.")

    if judgment is None or not judgment.get("answer_correct", False):
        return None

    return {
        "question": question,
        "question_type": question_type,
        "answer": judgment.get("verified_answer", answer),
        "trajectory": trajectory,
        "judgment": judgment,
        "seed_topic": seed_topic,
    }
