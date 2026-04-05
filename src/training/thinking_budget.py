"""Thinking budget processor — caps thinking and forces tool calls.

Two behaviors:
1. Caps <think> blocks at max_thinking_tokens
2. After ANY </think>, forces <tool_call> as the next token

This guarantees the model always calls a tool after thinking,
never produces a text response.
"""

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase


class ThinkingBudgetProcessor(LogitsProcessor):
    """Cap thinking at N tokens and force tool call after thinking ends."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_thinking_tokens: int = 256,
                 force_tool_after_think: bool = True):
        self.max_thinking_tokens = max_thinking_tokens
        self.force_tool_after_think = force_tool_after_think
        self.think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        self.think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.tool_call_id = tokenizer.encode("<tool_call>", add_special_tokens=False)[0]
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        self._in_think = {}
        self._think_tokens = {}
        self._just_ended_think = {}  # True = last token was </think> or \n after it

    def reset(self, assume_in_think: bool = True):
        """Reset state. If assume_in_think=True, start as if <think> was just generated.
        This is needed because we prepend <think> to the prompt, so the model
        starts generating inside a think block but the processor doesn't see
        the prepended token."""
        self._in_think = {}
        self._think_tokens = {}
        self._just_ended_think = {}
        self._assume_in_think = assume_in_think

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            last_token = input_ids[i, -1].item()

            # After </think>, strongly boost <tool_call> but don't force it
            if self._just_ended_think.get(i) == "need_newline":
                scores[i, self.newline_id] += 10.0
                self._just_ended_think[i] = "need_tool_call"
                continue
            elif self._just_ended_think.get(i) == "need_tool_call":
                scores[i, self.tool_call_id] += 10.0
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

            # Count and cap thinking tokens
            if self._in_think.get(i, False):
                self._think_tokens[i] = self._think_tokens.get(i, 0) + 1

                if self._think_tokens[i] >= self.max_thinking_tokens // 2:
                    # Soft nudge starting at 50% (128 tokens), progressively stronger
                    progress = (self._think_tokens[i] - self.max_thinking_tokens // 2) / (self.max_thinking_tokens // 2)
                    # Boost scales from +2 at 50% to +15 at 100%
                    boost = 2.0 + 13.0 * progress
                    scores[i, self.think_end_id] += boost

        return scores
