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
        assert "snippet" not in result  # search returns titles+URLs only, no snippets

    def test_search_caches_results(self):
        env = SearchEnvironment(provider=FakeProvider())
        r1 = env.search("cached query")
        r2 = env.search("cached query")
        assert r1 == r2

    def test_fetch_page_1(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/book", value="A" * 10000)

        env = SearchEnvironment(provider=FakeProvider(), page_size=100)
        result = env.fetch("https://example.com/book", page=1)
        assert result.startswith("A" * 100)
        assert "[Page 1 of 100]" in result

    def test_fetch_page_2(self):
        from src.env.search_env import _page_cache
        content = "A" * 100 + "B" * 100 + "C" * 100
        _page_cache.set("page", "https://example.com/pages", value=content)

        env = SearchEnvironment(provider=FakeProvider(), page_size=100)
        p2 = env.fetch("https://example.com/pages", page=2)
        assert p2.startswith("B" * 100)
        assert "[Page 2 of 3]" in p2

    def test_fetch_last_page(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/short", value="A" * 250)

        env = SearchEnvironment(provider=FakeProvider(), page_size=100)
        p3 = env.fetch("https://example.com/short", page=3)
        assert p3.startswith("A" * 50)
        assert "[Page 3 of 3]" in p3

    def test_fetch_invalid_page(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/tiny", value="hello")

        env = SearchEnvironment(provider=FakeProvider(), page_size=100)
        result = env.fetch("https://example.com/tiny", page=5)
        assert "[Invalid page 5" in result

    def test_fetch_default_page_is_1(self):
        from src.env.search_env import _page_cache
        _page_cache.set("page", "https://example.com/default", value="X" * 500)

        env = SearchEnvironment(provider=FakeProvider(), page_size=100)
        result = env.fetch("https://example.com/default")
        assert "[Page 1 of 5]" in result

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
