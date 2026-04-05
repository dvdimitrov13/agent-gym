"""Thinking budget processor — caps <think> blocks at N tokens.

Uses a LogitsProcessor to monitor thinking token count during generation.
After max_thinking_tokens, forces </think> to end the block so the model
can produce its actual output (tool calls).

Based on: https://muellerzr.github.io/til/end_thinking.html
"""

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase


class ThinkingBudgetProcessor(LogitsProcessor):
    """Force-ends <think> blocks after a token budget.

    Tracks tokens generated inside <think>...</think> blocks.
    Once the budget is exhausted, forces </think> token so the
    model can proceed to generate tool calls or text.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_thinking_tokens: int = 256):
        self.max_thinking_tokens = max_thinking_tokens
        self.think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        self.think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        # Per-sequence state
        self._in_think = {}        # seq_idx -> bool
        self._think_tokens = {}    # seq_idx -> count

    def reset(self):
        """Reset state for new generation batch."""
        self._in_think = {}
        self._think_tokens = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            last_token = input_ids[i, -1].item()

            # Track think block boundaries
            if last_token == self.think_start_id:
                self._in_think[i] = True
                self._think_tokens[i] = 0
            elif last_token == self.think_end_id:
                self._in_think[i] = False

            # Count tokens inside think block
            if self._in_think.get(i, False):
                self._think_tokens[i] = self._think_tokens.get(i, 0) + 1

                # Hard stop: force </think>
                if self._think_tokens[i] >= self.max_thinking_tokens:
                    scores[i, :] = float('-inf')
                    scores[i, self.think_end_id] = 0

                # Soft nudge in last 10%: boost </think> probability
                elif self._think_tokens[i] >= self.max_thinking_tokens * 0.9:
                    scores[i, self.think_end_id] += 5.0
                    scores[i, self.newline_id] += 2.0

        return scores
