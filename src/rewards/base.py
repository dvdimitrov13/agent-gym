"""Base classes for reward functions and reward composition.

RewardFunction — Protocol defining the callable signature for individual
reward components (compatible with TRL's GRPOTrainer).

RewardComposer — Composes multiple reward components using either
multiplicative or additive aggregation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions compatible with TRL's GRPOTrainer.

    Reward functions receive a batch of completions (each a list of
    message dicts from a multi-turn rollout) and return a score per
    completion.

    Dataset columns (answer, gold_passages, etc.) are forwarded as
    **kwargs by TRL.
    """

    def __call__(
        self,
        completions: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> list[float]:
        """Score a batch of completions.

        Args:
            completions: List of rollout message lists, one per generation.
            **kwargs: Dataset columns forwarded by TRL.

        Returns:
            List of reward scores, one per completion.
        """
        ...


class RewardComposer:
    """Composes multiple reward components into a single score.

    Supports two composition modes:

    - **multiplicative**: ``reward = primary × scaler_1 × scaler_2 × ...``
      Prevents the model from gaming one component to compensate for another.
      Backed by the GR³ paper (arxiv 2603.10535).

    - **additive**: ``reward = w_1 × r_1 + w_2 × r_2 + ...``
      Standard weighted sum. Simpler but allows compensatory shortcuts.

    Example::

        composer = RewardComposer(
            mode="multiplicative",
            primary=llm_judge_reward,
            scalers=[format_reward, thinking_multiplier_batch],
        )
        scores = composer(completions, **kwargs)

        # Or additive:
        composer = RewardComposer(
            mode="additive",
            components=[(llm_judge_reward, 1.0), (format_reward, 0.2)],
        )

    Args:
        mode: Composition mode — "multiplicative" or "additive".
        primary: The primary reward function (multiplicative mode only).
            Scalers are multiplied onto this.
        scalers: List of scaler functions that return values in [0, 1]
            (multiplicative mode only).
        components: List of (reward_fn, weight) tuples (additive mode only).
        floor: Minimum value for each scaler, preventing complete gradient
            death. E.g., floor=0.01 means ``max(scaler, 0.01)``.
    """

    def __init__(
        self,
        mode: str = "multiplicative",
        primary: RewardFunction | None = None,
        scalers: list[RewardFunction] | None = None,
        components: list[tuple[RewardFunction, float]] | None = None,
        floor: float = 0.0,
    ):
        if mode not in ("multiplicative", "additive"):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'multiplicative' or 'additive'.")

        self.mode = mode
        self.floor = floor

        if mode == "multiplicative":
            if primary is None:
                raise ValueError("multiplicative mode requires a primary reward function")
            self.primary = primary
            self.scalers = scalers or []
        else:
            if not components:
                raise ValueError("additive mode requires a non-empty components list")
            self.components = components

    def __call__(
        self,
        completions: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> list[float]:
        """Compute composed reward for a batch of completions.

        Args:
            completions: List of rollout message lists.
            **kwargs: Dataset columns forwarded by TRL.

        Returns:
            List of composed reward scores, one per completion.
        """
        if self.mode == "multiplicative":
            return self._multiplicative(completions, **kwargs)
        else:
            return self._additive(completions, **kwargs)

    def _multiplicative(
        self,
        completions: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> list[float]:
        """reward = primary × max(scaler_1, floor) × max(scaler_2, floor) × ..."""
        scores = self.primary(completions, **kwargs)

        for scaler_fn in self.scalers:
            scaler_values = scaler_fn(completions, **kwargs)
            scores = [
                s * max(v, self.floor)
                for s, v in zip(scores, scaler_values)
            ]

        return scores

    def _additive(
        self,
        completions: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> list[float]:
        """reward = w_1 × r_1 + w_2 × r_2 + ..."""
        n = len(completions)
        scores = [0.0] * n

        for reward_fn, weight in self.components:
            values = reward_fn(completions, **kwargs)
            scores = [s + weight * v for s, v in zip(scores, values)]

        return scores
