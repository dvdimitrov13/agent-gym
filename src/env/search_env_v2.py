"""SearchEnvironment with snippet IDs and submit_answer tool for v2 training.

Wraps the base SearchEnvironment to:
  - Add snippet IDs ([S1], [S2], [R1], etc.) to tool results
  - Provide submit_answer() tool for structured ranking output
  - Disable search/page caches for clean training signal

The snippet counter resets on each reset() call (per-episode).
"""

import re
from src.env.search_env import SearchEnvironment


class _NoCache:
    """Dummy cache that never hits."""
    def get(self, *args, **kwargs): return None
    def set(self, *args, **kwargs): pass


class SearchEnvironmentV2(SearchEnvironment):
    """SearchEnvironment with snippet IDs, submit_answer, and no cache."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._s_count = 0
        self._r_count = 0
        # Disable caches for clean training signal
        from src.env import search_env
        search_env._search_cache = _NoCache()
        search_env._page_cache = _NoCache()

    def reset(self, **kwargs) -> str | None:
        self._s_count = 0
        self._r_count = 0
        return super().reset(**kwargs)

    def search(self, query: str, max_results: int = 5) -> str:
        """Search the web. Returns snippets tagged with IDs like [S1], [S2].

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted search results with snippet IDs, titles, URLs, and snippets.
        """
        raw = super().search(query, max_results=max_results)

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
        """Read a web page and find matching sections. Returns excerpts tagged with IDs like [R1], [R2].

        Args:
            url: The URL to read.
            keywords: Keywords to search for within the page.

        Returns:
            Top matching paragraphs from the page with snippet IDs.
        """
        raw = super().read(url, keywords)

        if raw == "No matches found." or raw.startswith("[Error"):
            return raw

        lines = raw.split("\n")
        formatted = []
        for line in lines:
            m = re.match(r'^\[(\d+)\]$', line.strip())
            if m:
                self._r_count += 1
                formatted.append(f"[R{self._r_count}]")
            else:
                formatted.append(line)
        return "\n".join(formatted)

    def submit_answer(self, passage_ids: list[str]) -> str:
        """Submit the passages that answer the question, ordered by relevance. Call this when you have found the information needed to answer the question.

        Args:
            passage_ids: Ordered list of snippet IDs that answer the question (e.g. ["S3", "R1", "S1"]), most relevant first.

        Returns:
            Confirmation that the answer was submitted.
        """
        valid = [pid for pid in passage_ids if re.match(r'^[SR]\d+$', pid)]
        if not valid:
            return "Error: no valid passage IDs provided. Use IDs like S1, S2, R1, R2."
        return f"Answer submitted: {', '.join(valid)}"
