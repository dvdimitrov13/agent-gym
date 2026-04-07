"""ThinkingBudgetProcessor — LogitsProcessor that controls thinking length.

Modifies generation logits to:
  1. Soft-nudge ``</think>`` as thinking approaches the token budget
  2. Hard-force ``\\n`` immediately after ``</think>``
  3. Soft-boost ``<tool_call>`` after the newline

This guarantees the model transitions from thinking to tool calling,
preventing open-ended text responses.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase


class ThinkingBudgetProcessor(LogitsProcessor):
    """LogitsProcessor that caps thinking length and forces tool calls.

    After ``</think>``, forces a newline then soft-boosts ``<tool_call>``
    by +15 logits. During thinking, progressively boosts ``</think>``
    from +2 to +15 as the token count approaches the budget.

    Args:
        tokenizer: Tokenizer for resolving special token IDs.
        max_thinking_tokens: Hard budget for thinking tokens.
        force_tool_after_think: If True, force ``\\n`` + boost ``<tool_call>``
            after every ``</think>``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_thinking_tokens: int = 256,
        force_tool_after_think: bool = True,
    ) -> None:
        self.max_thinking_tokens = max_thinking_tokens
        self.soft_nudge_start = max_thinking_tokens // 2  # default: half of max
        self.force_tool_after_think = force_tool_after_think
        self.think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        self.think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.tool_call_id = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        self._in_think = {}
        self._think_tokens = {}
        self._just_ended_think = {}  # True = last token was </think> or \n after it

    def set_budget(self, soft_nudge_start: int, hard_cap: int) -> None:
        """Update thinking budget. Called by the curriculum callback."""
        self.soft_nudge_start = soft_nudge_start
        self.max_thinking_tokens = hard_cap

    def reset(self, assume_in_think: bool = True) -> None:
        """Reset per-sequence state for a new generation batch.

        Args:
            assume_in_think: If True, assume generation starts inside a
                ``<think>`` block (because we prepend it to the prompt).
        """
        self._in_think = {}
        self._think_tokens = {}
        self._just_ended_think = {}
        self._assume_in_think = assume_in_think

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            last_token = input_ids[i, -1].item()

            # After </think>, hard-force \n then soft boost <tool_call>
            if self._just_ended_think.get(i) == "need_newline":
                scores[i, :] = float('-inf')
                scores[i, self.newline_id] = 0
                self._just_ended_think[i] = "need_tool_call"
                continue
            elif self._just_ended_think.get(i) == "need_tool_call":
                scores[i, self.tool_call_id] += 15.0
                self._just_ended_think[i] = None
                continue

            # Track think block
            if last_token == self.think_start_id:
                self._in_think[i] = True
                self._think_tokens[i] = 0
            elif i not in self._in_think and self._assume_in_think:
                # First token for this sequence — assume we're in think
                # because <think> was prepended to the prompt
                self._in_think[i] = True
                self._think_tokens[i] = 0
            if last_token == self.think_end_id:
                self._in_think[i] = False
                if self.force_tool_after_think:
                    self._just_ended_think[i] = "need_newline"
                    continue

            # Count thinking tokens — soft nudge only, no hard cap
            # The reward multiplier handles length penalty (annealing to 0)
            if self._in_think.get(i, False):
                self._think_tokens[i] = self._think_tokens.get(i, 0) + 1

                if self._think_tokens[i] >= self.soft_nudge_start:
                    # Soft nudge: progressively boost </think> logit
                    remaining = self.max_thinking_tokens - self.soft_nudge_start
                    progress = (self._think_tokens[i] - self.soft_nudge_start) / max(remaining, 1)
                    # Boost scales from +2 at soft_nudge to +15 at max_thinking_tokens
                    # Beyond max_thinking_tokens, caps at +15
                    boost = min(2.0 + 13.0 * progress, 15.0)
                    scores[i, self.think_end_id] += boost

        return scores
