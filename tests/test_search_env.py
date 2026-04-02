import pytest
from src.env.search_env import SearchEnvironment
from src.env.providers.base import SearchProvider, SearchResult
from src.env.cache import SearchCache
from src.env.rate_limiter import RateLimiter


class FakeProvider(SearchProvider):
    """Deterministic provider for unit tests — no network calls."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"Result {i} for: {query}",
                url=f"https://example.com/{i}",
                snippet=f"This is snippet {i} about {query}.",
            )
            for i in range(1, min(max_results, 3) + 1)
        ]


class TestSearchEnvironment:
    def test_search_returns_formatted_results(self):
        env = SearchEnvironment(provider=FakeProvider())
        result = env.search("test query")
        assert "[1]" in result
        assert "example.com" in result
        assert "snippet" in result

    def test_search_caches_results(self):
        env = SearchEnvironment(provider=FakeProvider())
        r1 = env.search("cached query")
        r2 = env.search("cached query")
        assert r1 == r2

    def test_read_finds_keywords(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/read", value=(
            "Introduction to the topic.\n\n"
            "The capital of France is Paris. It is a major European city.\n\n"
            "Other facts about geography."
        ))

        env = SearchEnvironment(provider=FakeProvider())
        result = env.read("https://example.com/read", "Paris")
        assert "Paris" in result
        assert "capital of France" in result
        # Best match should be first
        assert result.index("capital of France") < result.index("---")

    def test_read_no_matches(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/nomatch", value="Nothing relevant here.")

        env = SearchEnvironment(provider=FakeProvider())
        result = env.read("https://example.com/nomatch", "Tokyo")
        assert result == "No matches found."

    def test_read_multiple_matches(self):
        from src.env.search_env import _page_cache
        content = "\n\n".join([
            f"Section {i}: The Rhine river flows through Germany."
            for i in range(10)
        ])
        _page_cache.set("page", "https://example.com/multi", value=content)

        env = SearchEnvironment(provider=FakeProvider())
        result = env.read("https://example.com/multi", "Rhine")
        assert "[1]" in result
        assert "Top 5" in result  # capped at 5

    def test_read_case_insensitive(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/case", value="The YANGTZE river is long.")

        env = SearchEnvironment(provider=FakeProvider())
        result = env.read("https://example.com/case", "yangtze")
        assert "YANGTZE" in result
        assert "Found 1 match" in result

    def test_reset_clears_episode_state(self):
        env = SearchEnvironment(provider=FakeProvider())
        env.search("query")
        assert env._search_count == 1
        env.reset()
        assert env._search_count == 0
        assert env._urls_seen == []


class TestCache:
    def test_memory_cache(self):
        cache = SearchCache()
        assert cache.get("a", "b") is None
        cache.set("a", "b", value="hello")
        assert cache.get("a", "b") == "hello"

    def test_disk_cache(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path / "cache")
        cache.set("x", "y", value="disk_value")
        cache2 = SearchCache(cache_dir=tmp_path / "cache")
        assert cache2.get("x", "y") == "disk_value"


class TestRateLimiter:
    def test_limiter_does_not_block_first_call(self):
        limiter = RateLimiter(requests_per_second=100)
        import time
        start = time.monotonic()
        limiter.wait()
        assert time.monotonic() - start < 0.05
