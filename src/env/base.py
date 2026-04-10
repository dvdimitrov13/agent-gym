"""Abstract base class for search environments.

Defines the contract that all search environments must implement.
TiToGRPOTrainer and the eval script depend on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSearchEnvironment(ABC):
    """Base class for search environments used in tool-calling RL.

    Subclasses must implement search(), read(), and submit_answer().
    The reset() method has a default no-op implementation that subclasses
    can override to clear per-episode state.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> str:
        """Search the web and return formatted results.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted search results as a string.
        """

    @abstractmethod
    def read(self, url: str, keywords: str) -> str:
        """Read a web page and return sections matching the given keywords.

        Args:
            url: The URL to read.
            keywords: Keywords to search for within the page.

        Returns:
            Matching excerpts from the page as a string.
        """

    def submit_answer(self, passage_ids: list[str]) -> str:
        """Submit passage IDs as the final answer.

        Args:
            passage_ids: Ordered list of snippet IDs (e.g. ["S3", "R1"]).

        Returns:
            Confirmation string.

        Raises:
            NotImplementedError: If the environment does not support submission.
        """
        raise NotImplementedError("This environment does not support submit_answer")

    def reset(self, **kwargs) -> str | None:
        """Reset per-episode state. Called between episodes.

        Override in subclasses to clear counters, URL tracking, etc.
        Caches should persist across resets.

        Returns:
            Optional initial observation string, or None.
        """
        return None
