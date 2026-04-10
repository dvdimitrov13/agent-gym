"""Protocol for reward functions.

Defines the callable signature that all reward functions must follow.
TRL's GRPOTrainer calls reward functions with (completions, **kwargs),
where kwargs contains dataset columns forwarded from the training data.

Using Protocol (structural typing) rather than ABC because reward
functions are standalone callables, not a class hierarchy.
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
