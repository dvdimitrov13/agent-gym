import os
import requests

from .base import SearchProvider, SearchResult
from ..rate_limiter import RateLimiter


class SerperProvider(SearchProvider):
    API_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str | None = None, rate_limit: float = 5.0):
        self._api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        if not self._api_key:
            raise ValueError("Serper API key required — set SERPER_API_KEY or pass api_key")
        self._limiter = RateLimiter(rate_limit)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self._limiter.wait()
        resp = requests.post(
            self.API_URL,
            json={"q": query, "num": max_results},
            headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
            )
            for r in resp.json().get("organic", [])[:max_results]
        ]
