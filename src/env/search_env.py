from src.config import CACHE_DIR, DEFAULT_SEARCH_MAX_RESULTS, DEFAULT_FETCH_MAX_CHARS
from src.env.cache import SearchCache
from src.env.extraction import fetch_and_extract
from src.env.providers.base import SearchProvider
from src.env.providers.duckduckgo import DuckDuckGoProvider

# Shared caches — persist across rollouts and episodes
_search_cache = SearchCache(cache_dir=CACHE_DIR / "search")
_page_cache = SearchCache(cache_dir=CACHE_DIR / "pages")


class SearchEnvironment:
    """Web search environment for TRL's environment_factory protocol.

    Public methods (search, fetch) are discovered by TRL via inspect and
    exposed as tools to the model. Type hints and docstrings are required —
    TRL uses them to generate tool schemas.
    """

    def __init__(self, provider: SearchProvider | None = None):
        self._provider = provider or DuckDuckGoProvider()
        self._search_count = 0
        self._fetch_count = 0
        self._urls_seen: list[str] = []

    def reset(self, **kwargs) -> str | None:
        """Reset per-episode state. Called by TRL between episodes.
        Caches persist — only per-episode counters and URL tracking reset."""
        self._search_count = 0
        self._fetch_count = 0
        self._urls_seen = []
        return None

    def search(self, query: str, max_results: int = DEFAULT_SEARCH_MAX_RESULTS) -> str:
        """Search the web and return a list of results with titles, URLs, and snippets.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted search results with titles, URLs, and snippets.
        """
        self._search_count += 1

        # Check cache
        cached = _search_cache.get("search", query, str(max_results))
        if cached:
            return cached

        # Live search
        try:
            results = self._provider.search(query, max_results=max_results)
        except Exception as e:
            return f"[Search error: {e}]"

        if not results:
            output = "[No results found]"
        else:
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] {r.title}")
                lines.append(f"    URL: {r.url}")
                lines.append(f"    {r.snippet}")
                lines.append("")
            output = "\n".join(lines).strip()

        _search_cache.set("search", query, str(max_results), value=output)
        return output

    def fetch(self, url: str, max_chars: int = DEFAULT_FETCH_MAX_CHARS, offset: int = 0) -> str:
        """Fetch and extract the text content of a web page.

        First call fetches the full page and caches it. Returns a window of
        max_chars characters starting at offset. If content is truncated,
        includes a note with remaining character count.

        Args:
            url: The URL to fetch.
            max_chars: Maximum characters to return in this window.
            offset: Character offset to start reading from.

        Returns:
            Extracted page content (Markdown when available, plain text otherwise).
        """
        self._fetch_count += 1
        if url not in self._urls_seen:
            self._urls_seen.append(url)

        # Get or fetch full page text
        full_text = _page_cache.get("page", url)
        if full_text is None:
            full_text = fetch_and_extract(url)
            _page_cache.set("page", url, value=full_text)

        # Apply windowing
        window = full_text[offset:offset + max_chars]
        remaining = len(full_text) - (offset + max_chars)

        if remaining > 0:
            window += f"\n\n[Content truncated. {remaining} characters remaining. Call fetch with offset={offset + max_chars} to continue.]"

        return window
