from src.config import CACHE_DIR, DEFAULT_SEARCH_MAX_RESULTS, FETCH_PAGE_SIZE
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

    fetch() works like reading pages in a book — each call returns the next
    ~500 words. The model decides whether to keep reading or move on.
    """

    def __init__(self, provider: SearchProvider | None = None, page_size: int = FETCH_PAGE_SIZE):
        self._provider = provider or DuckDuckGoProvider()
        self._page_size = page_size
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
        """Search the web and return a list of results with titles and URLs.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted search results with titles and URLs.
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
                lines.append(f"    {r.url}")
                lines.append("")
            output = "\n".join(lines).strip()

        _search_cache.set("search", query, str(max_results), value=output)
        return output

    def fetch(self, url: str, page: int = 1) -> str:
        """Read a specific page of content from a URL, like reading a book.

        Content is split into pages of ~500 words each. Page 1 is the
        beginning of the article, page 2 is the next section, and so on.

        Args:
            url: The URL to read.
            page: Which page to read (1-indexed). Defaults to 1.

        Returns:
            The requested page of content (~500 words).
        """
        self._fetch_count += 1
        if url not in self._urls_seen:
            self._urls_seen.append(url)

        # Get or fetch full page text
        full_text = _page_cache.get("page", url)
        if full_text is None:
            full_text = fetch_and_extract(url)
            _page_cache.set("page", url, value=full_text)

        total_pages = max(1, (len(full_text) + self._page_size - 1) // self._page_size)

        if page < 1 or page > total_pages:
            return f"[Invalid page {page}. This document has {total_pages} page(s).]"

        start = (page - 1) * self._page_size
        end = start + self._page_size
        chunk = full_text[start:end]

        chunk += f"\n\n[Page {page} of {total_pages}]"

        return chunk
