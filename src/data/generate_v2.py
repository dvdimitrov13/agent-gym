"""V2 data generator — retrieval-focused, 2025 events, snippet-ID ranking.

Changes from v1:
  - Uses OpenAI (GPT 5.4) instead of Anthropic
  - Questions focus on 2025+ events (post Qwen3 knowledge cutoff)
  - Tool results tagged with snippet IDs (S1, S2, R1, R2, ...)
  - Teacher model ranks snippets by relevance + source quality
  - Output includes gold_passages, sub_answers, and ranked snippet IDs
  - Expansion trick: some questions require read() for detail

Pipeline:
  1. Generate question about recent events
  2. Teacher searches with ID-tagged results
  3. Teacher ranks snippets
  4. Judge evaluates trajectory
  5. Save with ranking + sub_answers
"""

import json
import re
from openai import OpenAI
from src.env.search_env import SearchEnvironment


DEFAULT_MODEL = "gpt-5.4"


# --- Seed topics focused on 2025 events ---

SEED_TOPICS_2025 = [
    "2025 Nobel Prize winners and their work",
    "Major tech acquisitions and IPOs in 2025",
    "2025 Grammy, Oscar, and Emmy award winners",
    "Space missions launched in 2025",
    "2025 world leaders — elections, inaugurations, resignations",
    "Major international treaties and summits in 2025",
    "Natural disasters and extreme weather events in 2025",
    "Sports championships and world records set in 2025",
    "Major scientific discoveries published in 2025",
    "New UNESCO World Heritage Sites designated in 2025",
    "Company earnings milestones and stock market records in 2025",
    "Major product launches and tech announcements in 2025",
    "2025 conflicts, ceasefires, and peace agreements",
    "New laws and regulations that took effect in 2025",
    "Major infrastructure projects completed in 2025",
    "Celebrity and public figure deaths in 2025",
    "2026 events in the first quarter — January to April",
]


# --- Tool definitions (OpenAI format) ---

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web. Returns titled snippets with IDs like [S1], [S2], etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a web page and find sections matching keywords. "
                "Returns excerpts with IDs like [R1], [R2], etc. "
                "Use this when search snippets don't contain enough detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to read"},
                    "keywords": {"type": "string", "description": "Keywords to search for on the page"},
                },
                "required": ["url", "keywords"],
            },
        },
    },
]


# --- Prompts ---

QUESTION_EXAMPLES = {
    1: """\
Examples (these are about 2025 events — yours must also be about 2025+):
Hop 1: who won 2025 Grammy Album of the Year → [answer]
Question: Who won the Grammy Award for Album of the Year in 2025?

Hop 1: which country hosted the 2025 G20 summit → [answer]
Question: Which country hosted the G20 summit in 2025?""",
    2: """\
Examples (these are about 2025 events — yours must also be about 2025+):
Hop 1: who won 2025 Best Picture Oscar → [film name]
Hop 2: who directed [film name] → [director]
Question: Who directed the film that won Best Picture at the 2025 Academy Awards?

Hop 1: which company had the largest IPO in 2025 → [company]
Hop 2: who is the CEO of [company] → [person]
Question: Who is the CEO of the company that had the largest IPO in 2025?""",
    3: """\
Examples (these are about 2025 events — yours must also be about 2025+):
Hop 1: who won the 2025 Nobel Prize in Literature → [person]
Hop 2: where was [person] born → [city]
Hop 3: what country is [city] in → [country]
Question: In which country was the 2025 Nobel Prize in Literature winner born?

Hop 1: which spacecraft landed on Mars in 2025 → [mission name]
Hop 2: which space agency launched [mission] → [agency]
Hop 3: who is the head of [agency] → [person]
Question: Who heads the space agency that launched the Mars mission that landed in 2025?""",
}

HOP_FORMAT = {
    1: "Hop 1: [what to look up] → [answer]\nQuestion: [the question]",
    2: "Hop 1: [what to look up] → [answer]\nHop 2: [what to look up using hop 1] → [answer]\nQuestion: [the question]",
    3: "Hop 1: [what to look up] → [answer]\nHop 2: [what to look up using hop 1] → [answer]\nHop 3: [what to look up using hop 2] → [answer]\nQuestion: [the question]",
}

QUESTION_PROMPT = """\
Generate a question that requires chaining EXACTLY {num_hops} fact(s) together.
Topic area: {seed_topic}

CRITICAL RULES:
- The question MUST be about events, facts, or developments from 2025 or early 2026.
- Do NOT use historical or pre-2025 facts. Everything must be recent.
- The final answer must be SHORT (a name, number, date, or place).
- Each hop must depend on the previous hop's answer.
- The question must be unambiguous with exactly one correct answer.
- The facts must be verifiable through web search.

First, design the {num_hops}-hop chain, then write the question.

Format your response EXACTLY like this:
{hop_format}

{examples}"""

SEARCH_PROMPT = """\
Find information to answer this question using the search and read tools.

RULES:
- You MUST search the web. Do NOT answer from memory.
- Use search() to find relevant pages, then read(url, keywords) if snippets lack detail.
- Each tool result has an ID (like [S1], [R1]). Keep track of which snippets are useful.
- After searching, provide your ranking.

Question: {question}

After you have gathered enough information, respond with:
1. The answer (short — name, number, date, or place)
2. Your ranking of the most relevant snippets, ordered by relevance AND source quality.
   Prefer authoritative sources (official sites, major news outlets, Wikipedia) over blogs/forums.

Format your final response EXACTLY like this:
ANSWER: [short answer]
SUB_ANSWERS: [intermediate fact 1] | [intermediate fact 2] | ... | [final answer]
RANKING: [comma-separated snippet IDs in order of relevance, e.g. S3, R1, S1]
REASONING: [one line explaining why top-ranked snippets are best]"""

RETRY_SEARCH_PROMPT = """\
Find information to answer this question using the search and read tools.

RULES:
- You MUST search the web. Do NOT answer from memory.
- Use search() to find relevant pages, then read(url, keywords) if snippets lack detail.
- Each tool result has an ID (like [S1], [R1]). Keep track of which snippets are useful.

Question: {question}

A previous attempt had these issues:
{feedback}

Improve on the previous attempt. After gathering information, respond with:
ANSWER: [short answer]
SUB_ANSWERS: [intermediate fact 1] | [intermediate fact 2] | ... | [final answer]
RANKING: [comma-separated snippet IDs in order of relevance]
REASONING: [one line explaining why top-ranked snippets are best]"""

JUDGE_PROMPT = """\
You are evaluating a search trajectory for training an AI retrieval agent.
Judge ONLY based on the trajectory below — do not use external knowledge.

Question: {question}
Claimed answer: {answer}

The agent took the following search path:
{trajectory_summary}

Evaluate:
1. Does the answer appear in the retrieved content? This is the most important check.
2. Are the search queries well-formed and effective?
3. Does the ranking make sense? Are the most relevant, authoritative snippets ranked highest?
4. Were all hops covered? For multi-hop questions, each intermediate fact should be grounded.
5. Did the agent use read() when snippets were insufficient?

Respond with this JSON:
```json
{{
  "answer_correct": true/false,
  "answer_in_retrieved_content": true/false,
  "verified_answer": "short factual answer only",
  "ranking_quality": 1-5,
  "all_hops_covered": true/false,
  "needs_read": false,
  "overall_pass": true/false,
  "feedback": "specific tips for improvement"
}}
```

Set needs_read=true if the question clearly requires more detail than search snippets provide
(e.g., lists of items, specific statistics, detailed provisions of a law)."""

EXPAND_PROMPT = """\
You are making a search question HARDER by requiring a page read for specific detail.

Original question: {question}
Original answer: {answer}

Search results the agent found:
{trajectory_summary}

Look at the URLs above. Pick one result that likely contains specific details NOT visible
in the snippet (e.g., exact numbers, lists of items, specific dates, lesser-known names).

Rewrite the question to require reading that page for a specific detail.
The new answer must be a SHORT fact (name, number, date) that is NOT in the snippets.

Format:
Read URL: [URL to read]
Read keywords: [2-4 keywords]
New question: [extended question requiring page detail]
New answer: [short answer found only in the full page, not snippets]"""


# --- Snippet ID tracking ---

class SnippetTracker:
    """Assigns IDs to search/read results and reformats them."""

    def __init__(self):
        self.search_count = 0
        self.read_count = 0
        self.snippets = {}  # id -> {content, source_url, source_type}

    def format_search_results(self, raw_result: str) -> str:
        """Add S-IDs to search results.

        Input format (from SearchEnvironment):
            [1] Title
                URL
                Snippet text...

            [2] Title
                URL
                Snippet text...
        """
        lines = raw_result.split("\n")
        formatted = []
        current_sid = None
        current_url = None

        for line in lines:
            # Match result header: [1] Title
            m = re.match(r'^\[(\d+)\]\s+(.+)', line)
            if m:
                self.search_count += 1
                current_sid = f"S{self.search_count}"
                title = m.group(2).strip()
                formatted.append(f"[{current_sid}] {title}")
                self.snippets[current_sid] = {"content": "", "source_url": "", "type": "search", "title": title}
                current_url = None
                continue

            stripped = line.strip()
            if not stripped:
                formatted.append(line)
                current_sid = None
                current_url = None
                continue

            if current_sid:
                # URL line (starts with http)
                if stripped.startswith("http"):
                    current_url = stripped
                    self.snippets[current_sid]["source_url"] = current_url
                    formatted.append(f"    {stripped}")
                else:
                    # Snippet text
                    if self.snippets[current_sid]["content"]:
                        self.snippets[current_sid]["content"] += " " + stripped
                    else:
                        self.snippets[current_sid]["content"] = stripped
                    formatted.append(f"    {stripped}")
            else:
                formatted.append(line)

        return "\n".join(formatted)

    def format_read_results(self, raw_result: str, url: str) -> str:
        """Add R-IDs to read results."""
        # Read results are typically multi-paragraph
        # Split by double newline or numbered sections
        sections = re.split(r'\n\n+', raw_result)
        formatted = []

        for section in sections:
            section = section.strip()
            if not section or len(section) < 20:
                continue
            self.read_count += 1
            rid = f"R{self.read_count}"
            self.snippets[rid] = {"content": section, "source_url": url, "type": "read"}
            formatted.append(f"[{rid}] {url}")
            formatted.append(f"  {section}")
            formatted.append("")

        return "\n".join(formatted) if formatted else raw_result

    def get_snippet(self, sid: str) -> dict | None:
        return self.snippets.get(sid)

    def get_all_snippets(self) -> dict:
        return dict(self.snippets)


# --- Core pipeline ---

def dispatch_tool(env: SearchEnvironment, tracker: SnippetTracker, name: str, args: dict) -> str:
    """Execute a tool and format the result with snippet IDs."""
    if name == "search":
        raw = env.search(query=args["query"], max_results=args.get("max_results", 5))
        return tracker.format_search_results(raw)
    elif name == "read":
        raw = env.read(url=args["url"], keywords=args["keywords"])
        return tracker.format_read_results(raw, args["url"])
    return f"[Unknown tool: {name}]"


def run_with_tools(
    client: OpenAI,
    system: str,
    user_message: str,
    env: SearchEnvironment,
    tracker: SnippetTracker,
    model: str = DEFAULT_MODEL,
    max_rounds: int = 15,
    label: str = "",
) -> list[dict]:
    """Run a multi-turn conversation with tools via OpenAI API.
    Returns the full message history."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    prefix = f"    [{label}]" if label else "   "

    for round_num in range(1, max_rounds + 1):
        print(f"{prefix} round {round_num}...", flush=True)
        response = client.chat.completions.create(
            model=model, max_completion_tokens=4096,
            messages=messages, tools=TOOLS_OPENAI,
        )

        choice = response.choices[0]
        assistant_msg = {"role": "assistant", "content": choice.message.content}

        # Handle tool calls
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                args_short = json.dumps(args, ensure_ascii=False)[:100]
                print(f"{prefix}   → {tc.function.name}({args_short})", flush=True)

                result = dispatch_tool(env, tracker, tc.function.name, args)
                result_preview = result[:80].replace("\n", " ")
                print(f"{prefix}   ← {result_preview}...", flush=True)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # No tool calls — final response
            messages.append(assistant_msg)
            if choice.message.content:
                preview = choice.message.content[:100].replace("\n", " ")
                print(f"{prefix}   text: {preview}", flush=True)
            break

        if choice.finish_reason == "stop":
            break

    return messages


def extract_final_text(messages: list[dict]) -> str:
    """Get the final text response."""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def parse_ranking_response(text: str) -> dict:
    """Parse the teacher's final response for answer, sub_answers, ranking."""
    result = {"answer": "", "sub_answers": [], "ranking": [], "reasoning": ""}

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ANSWER:"):
            result["answer"] = line.split("ANSWER:", 1)[1].strip()
        elif line.startswith("SUB_ANSWERS:"):
            parts = line.split("SUB_ANSWERS:", 1)[1].strip()
            result["sub_answers"] = [s.strip() for s in parts.split("|") if s.strip()]
        elif line.startswith("RANKING:"):
            parts = line.split("RANKING:", 1)[1].strip()
            result["ranking"] = [s.strip() for s in parts.split(",") if s.strip()]
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split("REASONING:", 1)[1].strip()

    return result


def summarize_trajectory(messages: list[dict]) -> str:
    """Create a human-readable summary of the trajectory."""
    lines = []
    for msg in messages:
        if msg["role"] == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc["function"]
                    args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
                    args_str = json.dumps(args, ensure_ascii=False)
                    lines.append(f"  → {func['name']}({args_str})")
            elif msg.get("content"):
                preview = msg["content"][:200]
                lines.append(f"  response: {preview}")
        elif msg["role"] == "tool":
            content = msg.get("content", "")
            preview = content[:150] + "..." if len(content) > 150 else content
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
    client: OpenAI,
    seed_topic: str,
    model: str = DEFAULT_MODEL,
    num_hops: int = 2,
) -> str | None:
    """Step 1: Generate a question about 2025 events."""
    prompt = QUESTION_PROMPT.format(
        num_hops=num_hops,
        seed_topic=seed_topic,
        hop_format=HOP_FORMAT[num_hops],
        examples=QUESTION_EXAMPLES[num_hops],
    )
    response = client.chat.completions.create(
        model=model, max_completion_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()

    for line in text.split("\n"):
        if line.strip().startswith("Question:"):
            q = line.split("Question:", 1)[1].strip().strip('"').strip("'")
            return q if q and "?" in q else None
    return None


def step_search_trajectory(
    client: OpenAI,
    env: SearchEnvironment,
    tracker: SnippetTracker,
    question: str,
    feedback: str | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], dict]:
    """Step 2: Search and rank snippets.
    Returns (trajectory, parsed_response)."""
    env.reset()

    if feedback:
        prompt = RETRY_SEARCH_PROMPT.format(question=question, feedback=feedback)
    else:
        prompt = SEARCH_PROMPT.format(question=question)

    system = (
        "You are a research assistant evaluating web sources. "
        "Use tools to search the web. Each result has a snippet ID (S1, S2, R1, etc). "
        "After gathering information, rank the snippets by relevance AND source quality. "
        "Prefer authoritative sources: official sites > major news (Reuters, AP, BBC) > "
        "Wikipedia > specialized sites > blogs/forums."
    )

    messages = run_with_tools(
        client=client, system=system, user_message=prompt,
        env=env, tracker=tracker, model=model, label="search",
    )

    final_text = extract_final_text(messages)
    parsed = parse_ranking_response(final_text)
    return messages, parsed


def step_judge(
    client: OpenAI,
    question: str,
    answer: str,
    trajectory: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """Step 3: Judge evaluates trajectory quality."""
    summary = summarize_trajectory(trajectory)
    print("    [judge] evaluating...", flush=True)

    response = client.chat.completions.create(
        model=model, max_completion_tokens=2048,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, answer=answer, trajectory_summary=summary,
            ),
        }],
    )

    judge_text = response.choices[0].message.content.strip()
    return extract_json(judge_text)


def step_expand(
    client: OpenAI,
    env: SearchEnvironment,
    tracker: SnippetTracker,
    question: str,
    answer: str,
    trajectory: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """Step 4 (optional): Expand question to require a read() call."""
    summary = summarize_trajectory(trajectory)
    print("  Step 4: expanding question to require read()...", flush=True)

    response = client.chat.completions.create(
        model=model, max_completion_tokens=512,
        messages=[{
            "role": "user",
            "content": EXPAND_PROMPT.format(
                question=question, answer=answer, trajectory_summary=summary,
            ),
        }],
    )

    text = response.choices[0].message.content.strip()

    read_url = read_keywords = new_question = new_answer = None
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
    print(f"  Step 4 read: {read_url} [{read_keywords}]", flush=True)

    # Execute the read
    read_result = env.read(read_url, read_keywords)
    formatted_read = tracker.format_read_results(read_result, read_url)
    print(f"  Step 4 read result: {formatted_read[:100]}...", flush=True)

    if read_result == "No matches found.":
        print("  Step 4: read found nothing, skipping", flush=True)
        return None

    return {
        "question": new_question,
        "answer": new_answer,
        "expanded_from": question,
        "read_url": read_url,
        "read_keywords": read_keywords,
    }


def generate_training_example(
    client: OpenAI,
    env: SearchEnvironment,
    seed_topic: str,
    model: str = DEFAULT_MODEL,
    max_judge_retries: int = 2,
    num_hops: int = 2,
    try_expand: bool = True,
) -> dict | None:
    """Full pipeline: generate → search → rank → judge → (expand) → save."""

    # Step 1: Generate question
    print(f"  Step 1: generating {num_hops}-hop question about 2025...", flush=True)
    question = step_generate_question(client, seed_topic, model=model, num_hops=num_hops)
    if not question:
        print("  Step 1: failed", flush=True)
        return None
    print(f"  Step 1: {question}", flush=True)

    feedback = None
    trajectory = None
    parsed_response = None
    judgment = None
    tracker = SnippetTracker()

    for attempt in range(1 + max_judge_retries):
        # Step 2: Search + rank
        retry_note = f" (retry {attempt})" if feedback else ""
        print(f"  Step 2: search trajectory{retry_note}", flush=True)
        trajectory, parsed_response = step_search_trajectory(
            client, env, tracker, question, feedback=feedback, model=model,
        )
        if not parsed_response["answer"]:
            print("  Step 2: no answer produced", flush=True)
            return None
        print(f"  Step 2 answer: {parsed_response['answer']}", flush=True)
        print(f"  Step 2 ranking: {parsed_response['ranking']}", flush=True)
        print(f"  Step 2 sub_answers: {parsed_response['sub_answers']}", flush=True)

        # Step 3: Judge
        print("  Step 3: judging...", flush=True)
        judgment = step_judge(client, question, parsed_response["answer"], trajectory, model=model)
        if judgment is None:
            print("  Step 3: judge parse failed", flush=True)
            return None

        passed = judgment.get("overall_pass", False)
        print(f"  Step 3: pass={passed}", flush=True)

        if passed:
            break
        feedback = judgment.get("feedback", "Ensure answer is grounded in retrieved content.")

    if judgment is None or not judgment.get("overall_pass", False):
        return None

    # Check if trajectory used read()
    has_reads = any(
        tc["function"]["name"] == "read"
        for msg in trajectory if msg["role"] == "assistant" and msg.get("tool_calls")
        for tc in msg.get("tool_calls", [])
    )

    # Step 4 (optional): Expand to require read
    expanded = None
    if try_expand and not has_reads and judgment.get("needs_read", False):
        expanded = step_expand(
            client, env, tracker, question,
            parsed_response["answer"], trajectory, model=model,
        )

    # Build gold passages from ranking
    gold_passages = []
    for sid in parsed_response["ranking"]:
        snippet = tracker.get_snippet(sid)
        if snippet:
            gold_passages.append({
                "id": sid,
                "content": snippet["content"],
                "source_url": snippet["source_url"],
                "type": snippet["type"],
            })

    result = {
        "question": expanded["question"] if expanded else question,
        "num_hops": num_hops,
        "answer": expanded["answer"] if expanded else parsed_response["answer"],
        "sub_answers": parsed_response["sub_answers"],
        "gold_passages": gold_passages,
        "gold_ranking": parsed_response["ranking"],
        "all_snippets": tracker.get_all_snippets(),
        "trajectory": trajectory,
        "judgment": judgment,
        "seed_topic": seed_topic,
    }
    if expanded:
        result["expanded_from"] = expanded["expanded_from"]

    return result
