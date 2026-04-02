from ddgs import DDGS

from .base import SearchProvider, SearchResult
from ..rate_limiter import RateLimiter


class DuckDuckGoProvider(SearchProvider):
    def __init__(self, rate_limit: float = 1.0):
        self._limiter = RateLimiter(rate_limit)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self._limiter.wait()
        results = DDGS().text(query, max_results=max_results)
        return [
            SearchResult(title=r["title"], url=r["href"], snippet=r["body"])
            for r in results
        ]
