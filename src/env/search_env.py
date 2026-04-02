import re

from rapidfuzz import fuzz

from src.config import CACHE_DIR, DEFAULT_SEARCH_MAX_RESULTS
from src.env.cache import SearchCache
from src.env.extraction import fetch_and_extract
from src.env.providers.base import SearchProvider
from src.env.providers.duckduckgo import DuckDuckGoProvider

# Shared caches — persist across rollouts and episodes
_search_cache = SearchCache(cache_dir=CACHE_DIR / "search")
_page_cache = SearchCache(cache_dir=CACHE_DIR / "pages")

MAX_MATCHES = 5
CONTEXT_WORDS = 100
MIN_SCORE = 40  # minimum rapidfuzz score to count as a match


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines."""
    raw = re.split(r"\n{2,}", text)
    return [p.strip() for p in raw if p.strip()]


def _score_paragraph(paragraph: str, keywords: str) -> float:
    """Score a paragraph against keywords using rapidfuzz partial ratio."""
    return fuzz.partial_ratio(keywords.lower(), paragraph.lower())


def _trim_paragraph(paragraph: str) -> str:
    """Trim very long paragraphs to ~200 words."""
    words = paragraph.split()
    if len(words) <= CONTEXT_WORDS * 2:
        return paragraph
    return " ".join(words[:CONTEXT_WORDS]) + " [...] " + " ".join(words[-CONTEXT_WORDS:])


class SearchEnvironment:
    """Web search environment for TRL's environment_factory protocol.

    Two tools:
    - search(query) → titles, URLs, and snippets from the web
    - read(url, keywords) → fuzzy keyword search within a page, returns top matching paragraphs
    """

    def __init__(self, provider: SearchProvider | None = None):
        self._provider = provider or DuckDuckGoProvider()
        self._search_count = 0
        self._read_count = 0
        self._urls_seen: list[str] = []

    def reset(self, **kwargs) -> str | None:
        """Reset per-episode state. Called by TRL between episodes.
        Caches persist — only per-episode counters and URL tracking reset."""
        self._search_count = 0
        self._read_count = 0
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

        cached = _search_cache.get("search", query, str(max_results))
        if cached:
            return cached

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
                lines.append(f"    {r.snippet}")
                lines.append("")
            output = "\n".join(lines).strip()

        _search_cache.set("search", query, str(max_results), value=output)
        return output

    def read(self, url: str, keywords: str) -> str:
        """Read a web page and find sections matching the given keywords.

        Fetches the page (cached after first call), then finds the paragraphs
        that best match the keywords using fuzzy word-level matching.
        Returns up to 5 best matching excerpts.

        Args:
            url: The URL to read.
            keywords: Keywords to search for within the page.

        Returns:
            Top matching paragraphs from the page, or a message if no matches.
        """
        self._read_count += 1
        if url not in self._urls_seen:
            self._urls_seen.append(url)

        # Get or fetch full page text
        full_text = _page_cache.get("page", url)
        if full_text is None:
            full_text = fetch_and_extract(url)
            _page_cache.set("page", url, value=full_text)

        if full_text.startswith("[Error"):
            return full_text

        # Split into paragraphs and score each with rapidfuzz
        paragraphs = _split_paragraphs(full_text)
        if not paragraphs:
            return "No matches found."

        scored = [(para, _score_paragraph(para, keywords)) for para in paragraphs]
        scored = [(para, score) for para, score in scored if score >= MIN_SCORE]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return "No matches found."

        top = scored[:MAX_MATCHES]
        excerpts = [_trim_paragraph(para) for para, _ in top]

        header = f"Found {len(scored)} matching section(s). Top {len(excerpts)}:\n"
        body = "\n\n---\n\n".join(f"[{i+1}]\n{ex}" for i, ex in enumerate(excerpts))

        return header + body
