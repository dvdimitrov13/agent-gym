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

    def reset(self):
        self._in_think = {}
        self._think_tokens = {}
        self._just_ended_think = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            last_token = input_ids[i, -1].item()

            # State: just emitted </think> → force \n<tool_call>
            if self._just_ended_think.get(i) == "need_newline":
                scores[i, :] = float('-inf')
                scores[i, self.newline_id] = 0
                self._just_ended_think[i] = "need_tool_call"
                continue
            elif self._just_ended_think.get(i) == "need_tool_call":
                scores[i, :] = float('-inf')
                scores[i, self.tool_call_id] = 0
                self._just_ended_think[i] = None  # done
                continue

            # Track think block
            if last_token == self.think_start_id:
                self._in_think[i] = True
                self._think_tokens[i] = 0
            elif last_token == self.think_end_id:
                self._in_think[i] = False
                if self.force_tool_after_think:
                    self._just_ended_think[i] = "need_newline"
                    continue

            # Count and cap thinking tokens
            if self._in_think.get(i, False):
                self._think_tokens[i] = self._think_tokens.get(i, 0) + 1

                if self._think_tokens[i] >= self.max_thinking_tokens:
                    # Force </think>
                    scores[i, :] = float('-inf')
                    scores[i, self.think_end_id] = 0
                elif self._think_tokens[i] >= self.max_thinking_tokens * 0.9:
                    # Soft nudge
                    scores[i, self.think_end_id] += 5.0

        return scores
