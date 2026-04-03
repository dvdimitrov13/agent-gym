"""Tests for reward functions."""

from src.rewards import retrieval_reward, efficiency_reward, thinking_reward


# -- Helper: build a simple completion --

def _make_completion(text: str, tool_calls: list[dict] | None = None) -> list[dict]:
    """Build a minimal multi-turn completion for testing."""
    msgs = []
    if tool_calls:
        for tc in tool_calls:
            msgs.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use", "id": tc.get("id", "t1"),
                    "name": tc["name"], "input": tc["input"],
                }],
            })
            msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tc.get("id", "t1"),
                    "content": tc.get("result", "some result"),
                }],
            })
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return msgs


# === retrieval_reward ===

class TestRetrievalReward:
    def test_content_match_in_read(self):
        comp = [_make_completion(
            "done",
            tool_calls=[{
                "id": "r1", "name": "read",
                "input": {"url": "https://example.com", "keywords": "capital"},
                "result": "The capital of Baden-Württemberg is Stuttgart, a city in southern Germany.",
            }],
        )]
        assert retrieval_reward(comp, ["Stuttgart"]) == [1.0]

    def test_no_content_match(self):
        comp = [_make_completion(
            "done",
            tool_calls=[{
                "id": "r1", "name": "read",
                "input": {"url": "https://example.com", "keywords": "capital"},
                "result": "Berlin is the capital of Germany.",
            }],
        )]
        assert retrieval_reward(comp, ["Stuttgart"]) == [0.0]

    def test_url_recall(self):
        comp = [_make_completion(
            "done",
            tool_calls=[{
                "id": "s1", "name": "search",
                "input": {"query": "test"},
                "result": "[1] Result\n    https://example.com/page1\n    snippet",
            }],
        )]
        result = retrieval_reward(
            comp, ["yes"],
            gold_urls=[["https://example.com/page1", "https://example.com/page2"]],
        )
        # Found 1/2 gold URLs = 0.5 url_score, no read results = 0.0 content
        # 0.5 * 0.5 + 0.5 * 0.0 = 0.25
        assert result == [0.25]

    def test_no_tools_no_reward(self):
        comp = [_make_completion("Paris")]
        assert retrieval_reward(comp, ["Paris"]) == [0.0]

    def test_answer_in_snippet(self):
        """Answer found in search snippet counts as retrieval success."""
        comp = [_make_completion(
            "done",
            tool_calls=[{
                "id": "s1", "name": "search",
                "input": {"query": "NAFTA members"},
                "result": "[1] NAFTA\n    https://example.com\n    NAFTA had 3 original member countries.",
            }],
        )]
        # Answer "3" appears in snippet — should get content match
        assert retrieval_reward(comp, ["3"]) == [1.0]


# === efficiency_reward ===

class TestEfficiencyReward:
    def test_same_steps(self):
        comp = [_make_completion(
            "done",
            tool_calls=[
                {"name": "search", "input": {"query": "q1"}},
                {"name": "read", "input": {"url": "u1", "keywords": "k1"}},
            ],
        )]
        assert efficiency_reward(comp, gold_tool_count=[2]) == [1.0]

    def test_fewer_steps(self):
        comp = [_make_completion(
            "done",
            tool_calls=[{"name": "search", "input": {"query": "q1"}}],
        )]
        assert efficiency_reward(comp, gold_tool_count=[3]) == [1.0]

    def test_double_steps(self):
        comp = [_make_completion(
            "done",
            tool_calls=[
                {"id": "t1", "name": "search", "input": {"query": "q1"}},
                {"id": "t2", "name": "search", "input": {"query": "q2"}},
                {"id": "t3", "name": "search", "input": {"query": "q3"}},
                {"id": "t4", "name": "read", "input": {"url": "u1", "keywords": "k"}},
            ],
        )]
        assert efficiency_reward(comp, gold_tool_count=[2]) == [0.0]

    def test_one_extra(self):
        comp = [_make_completion(
            "done",
            tool_calls=[
                {"id": "t1", "name": "search", "input": {"query": "q1"}},
                {"id": "t2", "name": "search", "input": {"query": "q2"}},
                {"id": "t3", "name": "read", "input": {"url": "u1", "keywords": "k"}},
            ],
        )]
        assert efficiency_reward(comp, gold_tool_count=[2]) == [0.5]

    def test_no_gold(self):
        comp = [_make_completion("done")]
        assert efficiency_reward(comp) == [1.0]

    def test_zero_tools_when_expected(self):
        comp = [_make_completion("I know the answer already")]
        assert efficiency_reward(comp, gold_tool_count=[3]) == [0.0]


# === thinking_reward ===

class TestThinkingReward:
    def test_short_thinking(self):
        think = "<think>I need to search for the capital of France.</think>\nParis"
        comp = [_make_completion(think)]
        assert thinking_reward(comp) == [1.0]

    def test_no_thinking(self):
        comp = [_make_completion("Just the answer")]
        assert thinking_reward(comp) == [0.5]

    def test_long_thinking(self):
        # 200+ words of rambling
        words = " ".join(["word"] * 250)
        think = f"<think>{words}</think>\nanswer"
        comp = [_make_completion(think)]
        assert thinking_reward(comp) == [0.0]

    def test_medium_thinking(self):
        # ~125 words — midpoint between 50 and 200
        words = " ".join(["word"] * 125)
        think = f"<think>{words}</think>\nanswer"
        comp = [_make_completion(think)]
        result = thinking_reward(comp)
        assert 0.4 < result[0] < 0.6  # should be ~0.5

    def test_multiple_think_blocks(self):
        """Total words across all think blocks counts."""
        comp = [_make_completion("final")]
        # Add two assistant messages with think blocks
        words30 = " ".join(["word"] * 30)
        comp[0] = {
            "role": "assistant",
            "content": [{"type": "text", "text": f"<think>{words30}</think>"}],
        }
        # Insert another think block as a second assistant message
        comp.insert(0, {
            "role": "assistant",
            "content": [{"type": "text", "text": f"<think>{words30}</think>"}],
        })
        result = thinking_reward([comp])
        # 60 total words — just above short_threshold (50), should be close to 1.0
        assert result[0] > 0.9
