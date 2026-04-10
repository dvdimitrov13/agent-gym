"""Web search environment for tool-calling RL.

Three tools:
- search(query) → web search results tagged with snippet IDs [S1], [S2], etc.
- read(url, keywords) → fuzzy keyword search within a page, tagged [R1], [R2], etc.
- submit_answer(passage_ids) → submit retrieved passages as the final answer
"""

from __future__ import annotations

import logging
import re
import time

from rapidfuzz import fuzz

from src.config import CACHE_DIR, DEFAULT_SEARCH_MAX_RESULTS
from src.env.base import BaseSearchEnvironment
from src.env.cache import SearchCache
from src.env.extraction import fetch_and_extract
from src.env.providers.base import SearchProvider
from src.env.providers.duckduckgo import DuckDuckGoProvider

logger = logging.getLogger(__name__)

# Shared caches — persist across rollouts and episodes
_search_cache: SearchCache | _NoCache = SearchCache(cache_dir=CACHE_DIR / "search")
_page_cache: SearchCache | _NoCache = SearchCache(cache_dir=CACHE_DIR / "pages")

MAX_MATCHES = 5
CONTEXT_WORDS = 100
MIN_SCORE = 40  # minimum rapidfuzz score to count as a match


class _NoCache:
    """Dummy cache that never hits — used during training for clean signal."""

    def get(self, *args, **kwargs) -> None:
        return None

    def set(self, *args, **kwargs) -> None:
        pass


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


class SearchEnvironment(BaseSearchEnvironment):
    """Web search environment for TRL's environment_factory protocol.

    Results are tagged with snippet IDs ([S1], [S2] for search, [R1], [R2]
    for read) so the model can reference specific passages when submitting.

    Args:
        provider: Search provider to use. Defaults to DuckDuckGo.
        use_cache: Whether to cache search/page results. Disable during
            training for clean reward signal; enable during data generation.
    """

    def __init__(
        self,
        provider: SearchProvider | None = None,
        use_cache: bool = True,
    ):
        self._provider = provider or DuckDuckGoProvider()
        self._search_count = 0
        self._read_count = 0
        self._urls_seen: list[str] = []
        self._s_count = 0
        self._r_count = 0

        if not use_cache:
            global _search_cache, _page_cache
            _search_cache = _NoCache()
            _page_cache = _NoCache()

    def reset(self, **kwargs) -> str | None:
        """Reset per-episode state. Called by TRL between episodes.

        Caches persist — only per-episode counters and URL tracking reset.
        """
        self._search_count = 0
        self._read_count = 0
        self._urls_seen = []
        self._s_count = 0
        self._r_count = 0
        return None

    def search(self, query: str, max_results: int = DEFAULT_SEARCH_MAX_RESULTS) -> str:
        """Search the web. Returns snippets tagged with IDs like [S1], [S2].

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted search results with snippet IDs, titles, URLs, and snippets.
        """
        self._search_count += 1
        t0 = time.time()

        cached = _search_cache.get("search", query, str(max_results))
        if cached:
            logger.info(f"search('{query[:50]}') cache hit ({time.time()-t0:.2f}s)")
            # Re-tag cached results with current snippet counter
            return self._tag_search_results(cached)

        try:
            results = self._provider.search(query, max_results=max_results)
        except Exception as e:
            logger.warning(f"search('{query[:50]}') error: {e} ({time.time()-t0:.2f}s)")
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
        logger.info(f"search('{query[:50]}') {len(results) if results else 0} results ({time.time()-t0:.2f}s)")
        return self._tag_search_results(output)

    def _tag_search_results(self, raw: str) -> str:
        """Replace numeric IDs [1], [2] with snippet IDs [S1], [S2]."""
        lines = raw.split("\n")
        formatted = []
        for line in lines:
            m = re.match(r'^\[(\d+)\]\s+(.*)', line)
            if m:
                self._s_count += 1
                formatted.append(f"[S{self._s_count}] {m.group(2)}")
            else:
                formatted.append(line)
        return "\n".join(formatted)

    def read(self, url: str, keywords: str) -> str:
        """Read a web page and find matching sections. Returns excerpts tagged [R1], [R2].

        Fetches the page (cached after first call), then finds the paragraphs
        that best match the keywords using fuzzy word-level matching.
        Returns up to 5 best matching excerpts.

        Args:
            url: The URL to read.
            keywords: Keywords to search for within the page.

        Returns:
            Top matching paragraphs from the page with snippet IDs.
        """
        self._read_count += 1
        t0 = time.time()
        if url not in self._urls_seen:
            self._urls_seen.append(url)

        # Get or fetch full page text
        full_text = _page_cache.get("page", url)
        if full_text is None:
            full_text = fetch_and_extract(url)
            _page_cache.set("page", url, value=full_text)
            logger.info(f"read('{url[:60]}') fetched page ({time.time()-t0:.2f}s)")

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

        # Tag with R-prefixed snippet IDs
        tagged = []
        for ex in excerpts:
            self._r_count += 1
            tagged.append(f"[R{self._r_count}]\n{ex}")

        header = f"Found {len(scored)} matching section(s). Top {len(excerpts)}:\n"
        body = "\n\n---\n\n".join(tagged)

        logger.info(f"read('{url[:60]}', '{keywords[:30]}') {len(scored)} matches ({time.time()-t0:.2f}s)")
        return header + body

    def submit_answer(self, passage_ids: list[str]) -> str:
        """Submit the passages that answer the question, ordered by relevance.

        Call this when you have found the information needed to answer the
        question. Immediately ends the trajectory.

        Args:
            passage_ids: Ordered list of snippet IDs that answer the question
                (e.g. ["S3", "R1", "S1"]), most relevant first.

        Returns:
            Confirmation that the answer was submitted.
        """
        valid = [pid for pid in passage_ids if re.match(r'^[SR]\d+$', pid)]
        if not valid:
            return "Error: no valid passage IDs provided. Use IDs like S1, S2, R1, R2."
        return f"Answer submitted: {', '.join(valid)}"
