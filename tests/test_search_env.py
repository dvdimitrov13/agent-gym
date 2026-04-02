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

    def test_fetch_with_windowing(self):
        env = SearchEnvironment(provider=FakeProvider())
        # Manually populate page cache to avoid network calls
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/test", value="A" * 10000)

        result = env.fetch("https://example.com/test", max_chars=100, offset=0)
        assert len(result.split("\n\n[Content truncated")[0]) == 100
        assert "characters remaining" in result

    def test_fetch_offset_continues(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/offset", value="ABCDE" * 100)

        env = SearchEnvironment(provider=FakeProvider())
        r1 = env.fetch("https://example.com/offset", max_chars=10, offset=0)
        assert r1.startswith("ABCDEABCDE")

        r2 = env.fetch("https://example.com/offset", max_chars=10, offset=10)
        assert r2.startswith("ABCDEABCDE")

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
        # New cache instance reads from disk
        cache2 = SearchCache(cache_dir=tmp_path / "cache")
        assert cache2.get("x", "y") == "disk_value"


class TestRateLimiter:
    def test_limiter_does_not_block_first_call(self):
        limiter = RateLimiter(requests_per_second=100)
        import time
        start = time.monotonic()
        limiter.wait()
        assert time.monotonic() - start < 0.05
