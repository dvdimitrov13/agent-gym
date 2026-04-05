"""TiToGRPOTrainer — single-GPU trainer with token-space tool calling.

Overrides TRL's _tool_call_loop to splice tool results in token space
instead of re-templating (TI/TO approach from SID-1).

Key behaviors:
  - submit_answer immediately terminates a trajectory (no extra generation)
  - No tool call = trajectory terminated (agent must always use tools)
  - Thinking tokens stripped from carried-forward context
  - GPU memory logged each round
  - Only NEW tokens searched for tool calls (prevents re-detection bug)
"""

import json
import logging
import os
import time

import torch
from trl import GRPOTrainer
from transformers import TrainerCallback

from src.training.tito import (
    _init_token_ids, _find_tool_call, _parse_tool_call_json,
    _encode_tool_result, strip_thinking_tokens,
    _TOOL_CALL_END_ID,
)

logger = logging.getLogger(__name__)


def _find_tc_end(cids: list[int]) -> int:
    """Find position after </tool_call> in token list."""
    for j in range(len(cids) - 1, -1, -1):
        if cids[j] == _TOOL_CALL_END_ID:
            return j + 1
    return len(cids)


class TiToGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with token-space tool calling (TI/TO)."""

    def __init__(self, *args, thinking_budget: int = 256, **kwargs):
        # Remove custom kwargs before passing to parent
        kwargs.pop("force_submit_until_step", None)
        kwargs.pop("disable_thinking", None)
        super().__init__(*args, **kwargs)
        _init_token_ids(self.processing_class)

        # Thinking budget: cap <think> blocks at N tokens via LogitsProcessor
        self.thinking_budget = thinking_budget
        if thinking_budget > 0:
            from src.training.thinking_budget import ThinkingBudgetProcessor
            self._thinking_processor = ThinkingBudgetProcessor(
                self.processing_class, max_thinking_tokens=thinking_budget,
            )
            logger.info(f"TiToGRPOTrainer: TI/TO enabled, thinking budget={thinking_budget} tokens")
        else:
            self._thinking_processor = None
            logger.info("TiToGRPOTrainer: TI/TO enabled, no thinking budget")

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Override to apply thinking multiplier as a scalar on all rewards."""
        import torch
        from src.rewards.thinking_reward import thinking_multiplier

        # Let TRL compute rewards normally
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply thinking multiplier to each rollout's total
        for i, completion in enumerate(completions):
            mult = thinking_multiplier(completion)
            rewards_per_func[i, :] *= mult

        return rewards_per_func

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Override to inject thinking budget processor into generation.

        TRL passes generation_config to model.generate(), which doesn't accept
        logits_processor. We wrap the model's generate method to inject it.
        """
        if self._thinking_processor is not None:
            self._thinking_processor.reset()
            processor = self._thinking_processor

            # Temporarily wrap model.generate to inject logits_processor
            import types
            unwrapped = self.model
            while hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module
            if hasattr(unwrapped, 'base_model'):  # PEFT
                base = unwrapped.base_model
                if hasattr(base, 'model'):
                    base = base.model
            else:
                base = unwrapped

            original_generate = base.generate
            think_prefix_ids = self.processing_class.encode("<think>\n", add_special_tokens=False)

            def patched_generate(*args, **kwargs):
                existing = kwargs.get('logits_processor', [])
                if not isinstance(existing, list):
                    existing = list(existing) if existing else []
                processor.reset(assume_in_think=True)
                kwargs['logits_processor'] = existing + [processor]

                # Prepend <think>\n to input_ids to guarantee thinking mode
                inp = kwargs.get('input_ids', args[0] if args else None)
                if inp is not None:
                    prefix = torch.tensor([think_prefix_ids], device=inp.device).expand(inp.shape[0], -1)
                    kwargs['input_ids'] = torch.cat([inp, prefix], dim=1)
                    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                        prefix_mask = torch.ones(inp.shape[0], len(think_prefix_ids),
                                                 device=kwargs['attention_mask'].device, dtype=kwargs['attention_mask'].dtype)
                        kwargs['attention_mask'] = torch.cat([kwargs['attention_mask'], prefix_mask], dim=1)
                    if args:
                        args = args[1:]  # remove input_ids from positional args

                return original_generate(*args, **kwargs)

            base.generate = patched_generate
            try:
                result = super()._generate_single_turn(prompt_ids, images, multimodal_fields)
            finally:
                base.generate = original_generate
            return result
        return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions,
                        logprobs, images, multimodal_fields):
        """Token-space tool calling loop.

        Rules:
          - submit_answer → execute, STOP trajectory immediately
          - search/read → execute, splice result, generate next turn
          - no tool call → STOP trajectory (agent must always use tools)
          - Only searches NEW tokens for tool calls (not old spliced content)
        """
        tokenizer = self.processing_class
        device = next(self.model.parameters()).device
        tool_call_count = 0
        tool_failure_count = 0

        tool_mask_list = [[1] * len(cids) for cids in completion_ids]

        # Track active completions and where new tokens start per completion
        active = set(range(len(completion_ids)))
        new_tokens_start = {idx: 0 for idx in range(len(completion_ids))}
        total_submitted = 0
        total_no_tool = 0

        for iteration in range(self.max_tool_calling_iterations):
            if not active:
                break

            # Classify each active completion's tool call (NEW tokens only)
            submitted = []
            searching = []
            no_tool = []

            for idx in list(active):
                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                search_start = new_tokens_start.get(idx, 0)
                new_portion = cids[search_start:]
                span = _find_tool_call(new_portion)
                if span:
                    span = (span[0] + search_start, span[1] + search_start)
                if not span:
                    no_tool.append(idx)
                    continue
                parsed = _parse_tool_call_json(tokenizer, cids, span[0], span[1])
                if not parsed:
                    no_tool.append(idx)
                    continue
                name, args = parsed
                if name == "submit_answer":
                    submitted.append((idx, name, args))
                else:
                    searching.append((idx, name, args))

            # 1. submit_answer → execute and STOP
            for idx, name, args in submitted:
                tool_call_count += 1
                result = self._dispatch_tito_tool(name, args)
                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                tc_end = _find_tc_end(cids[new_tokens_start.get(idx, 0):]) + new_tokens_start.get(idx, 0)
                kept = cids[:tc_end]
                splice = _encode_tool_result(tokenizer, result)
                tool_mask_list[idx] = tool_mask_list[idx][:tc_end] + [0] * len(splice)
                completion_ids[idx] = kept + splice
                active.discard(idx)
                total_submitted += 1
                ids_str = args.get("passage_ids", [])
                logger.info(f"TI/TO [{idx}] submit_answer({ids_str})")

            # 2. No tool call → terminate
            for idx in no_tool:
                active.discard(idx)
            total_no_tool += len(no_tool)
            if no_tool:
                logger.info(f"TI/TO iter {iteration}: {len(no_tool)} completions terminated (no tool call)")

            if not active:
                break

            # 3. search/read → execute, splice, prepare next generation
            gen_prompts = []
            gen_idxs = []

            for idx, name, args in searching:
                if idx not in active:
                    continue
                tool_call_count += 1
                args_short = json.dumps(args)[:80]
                logger.info(f"TI/TO [{idx}] {name}({args_short})")
                result = self._dispatch_tito_tool(name, args)

                cids = completion_ids[idx] if isinstance(completion_ids[idx], list) else completion_ids[idx].tolist()
                tc_end = _find_tc_end(cids[new_tokens_start.get(idx, 0):]) + new_tokens_start.get(idx, 0)
                kept = cids[:tc_end]
                splice = _encode_tool_result(tokenizer, result)

                tool_mask_list[idx] = tool_mask_list[idx][:tc_end] + [0] * len(splice)
                completion_ids[idx] = kept + splice

                pid = prompt_ids[idx] if isinstance(prompt_ids[idx], list) else prompt_ids[idx].tolist()
                ctx = strip_thinking_tokens(kept)
                gen_prompts.append(pid + ctx + splice)
                gen_idxs.append(idx)

            # 4. Batch generate next turn
            if gen_prompts:
                new_tokens_list = self._batch_generate(gen_prompts, device, tokenizer)

                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9

                for i, idx in enumerate(gen_idxs):
                    new_tokens = new_tokens_list[i]
                    new_tokens_start[idx] = len(completion_ids[idx])
                    completion_ids[idx] = completion_ids[idx] + new_tokens
                    tool_mask_list[idx] = tool_mask_list[idx] + [1] * len(new_tokens)
                    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    completions[idx].append({"role": "assistant", "content": new_text})

                logger.info(f"TI/TO round {iteration+1}: {len(gen_idxs)} continuations "
                            f"(GPU: {mem_used:.1f}GB alloc, {mem_reserved:.1f}GB reserved)")

        avg_len = sum(len(c) if isinstance(c, list) else c.shape[0] for c in completion_ids) / len(completion_ids)
        logger.info(f"TI/TO done: {total_submitted} submitted, {total_no_tool} no-tool, "
                     f"{len(active)} still active, {tool_call_count} tool calls, "
                     f"avg completion={avg_len:.0f} tokens")

        return tool_mask_list, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _batch_generate(self, prompts, device, tokenizer):
        """Batch generate from token ID prompts. Returns new tokens per prompt.

        Prepends <think> to each prompt to guarantee thinking mode entry.
        ThinkingBudgetProcessor then caps thinking and forces <tool_call>.
        """
        # Prepend <think>\n to each prompt so model enters thinking mode
        think_prefix = tokenizer.encode("<think>\n", add_special_tokens=False)
        prompts_with_think = [p + think_prefix for p in prompts]

        max_len = max(len(p) for p in prompts_with_think)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        padded = []
        attn_masks = []
        for p in prompts_with_think:
            padding = [pad_id] * (max_len - len(p))
            padded.append(padding + p)
            attn_masks.append([0] * len(padding) + [1] * len(p))

        input_ids = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(attn_masks, device=device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_completion_length,
            temperature=self.args.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pad_id,
        )
        if self._thinking_processor is not None:
            self._thinking_processor.reset(assume_in_think=True)
            gen_kwargs["logits_processor"] = [self._thinking_processor]

        # Must set eval mode during generation — LoRA dropout in train mode
        # causes degenerate output. TRL does this via unwrap_model_for_generation.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
        if was_training:
            self.model.train()

        results = []
        for i in range(len(prompts_with_think)):
            new_tokens = outputs[i, max_len:].tolist()
            while new_tokens and new_tokens[-1] == pad_id:
                new_tokens.pop()
            # Prepend think_prefix to returned tokens so it's part of the completion
            results.append(list(think_prefix) + new_tokens)
        return results

    def _dispatch_tito_tool(self, name, args):
        """Dispatch a tool call."""
        if self.tools:
            for tool in self.tools:
                if getattr(tool, '__name__', None) == name:
                    try:
                        return str(tool(**args))
                    except Exception as e:
                        return f"Error: {e}"

        from src.env.search_env_v2 import SearchEnvironmentV2
        env = SearchEnvironmentV2()
        if name == "search":
            return env.search(**args)
        elif name == "read":
            return env.read(**args)
        elif name == "submit_answer":
            return env.submit_answer(**args)
        return f"Unknown tool: {name}"


class TrajectoryLoggingCallback(TrainerCallback):
    """Log a full decoded trajectory every N steps for diagnostics."""

    def __init__(self, every_n_steps=10, log_dir="/root/trajectories"):
        self.every_n_steps = every_n_steps
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        log_path = os.path.join(self.log_dir, f"step_{step:04d}.json")
        log_data = {"step": step, "metrics": {k: str(v) for k, v in (logs or {}).items()}}
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Trajectory log saved: {log_path}")
