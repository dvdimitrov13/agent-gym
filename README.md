# Agent Gym

RL-trained web search agent using GRPO. Teaches a language model to search the web, read pages, and retrieve information through reinforcement learning — no supervised fine-tuning on tool-use demonstrations.

Built on [TRL](https://github.com/huggingface/trl) with Qwen3-14B + LoRA, trained on 2x A6000 GPUs.

## Motivation

Inspired by the [SID-1 Technical Report](https://www.sid.ai/research/sid-1-technical-report), which showed that RL alone can teach models effective web search behavior. The key insight: rather than supervising *how* to search, reward *what* was found — the model discovers search strategies on its own.

We adapt this to a minimal two-tool setup (`search` + `fetch`) and focus on retrieval quality as the primary training signal.

### How we diverge from SID-1

SID-1 uses **Magistral's modified GRPO** with several important design choices we don't yet replicate:

| Aspect | SID-1 | Agent Gym (current) |
|--------|-------|-------------------|
| RL algorithm | Magistral's modified GRPO | Vanilla GRPO via TRL |
| Length bias | **Kept** — they prove removing it causes OOV token collapse | Removed (DAPO) — may cause instability on longer runs |
| Reward | **NDCG** over ranked document lists — partial credit, diminishing returns per rank | Binary content match — answer in tool results or not |
| Token handling | **TI/TO** — raw token sequences only, never convert to messages | TRL handles message templating (may introduce instabilities SID-1 warns about) |
| Length scheduling | Start short, gradually increase max rollout length | Fixed 1024 throughout |
| Format reward | Added later to prevent tool-calling regression | Not used |

These are known gaps. The DAPO vs length-biased GRPO choice and the binary vs NDCG reward are the highest priority to revisit.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
│                                                           │
│   GPU 0: Inference Server          GPU 1: TRL Trainer     │
│   ┌─────────────────────┐         ┌────────────────────┐  │
│   │ FastAPI + transformers│◄──────►│ RemoteGRPOTrainer  │  │
│   │ model.generate()      │  HTTP  │ GRPO loss + LoRA   │  │
│   │ LoRA hot-reload       │        │ gradient updates   │  │
│   └─────────────────────┘         └────────────────────┘  │
│            ▲                              │                │
│            │ POST /update_weights         │                │
│            └──────────────────────────────┘                │
│                   (LoRA sync after each step)              │
└─────────────────────────────────────────────────────────┘
```

**Why this setup?** vLLM requires `transformers<5` but TRL's `environment_factory` (which handles tool-calling RL) requires `transformers>=5.2`. Rather than fight version conflicts, we built a lightweight FastAPI inference server that uses plain `transformers` on one GPU while TRL trains on the other. Inspired by [PipelineRL](https://github.com/ServiceNow/PipelineRL) (used by OLMo-3) which decouples generation from training, but without the Redis/Ray complexity.

This gives us ~2 min/step vs ~5 min with single-GPU sequential generation.

## How It Works

### Tools

The agent has two tools, following SID-1's two-tier retrieval hierarchy:

- **`search(query)`** — Web search via DuckDuckGo. Returns snippets from multiple results. Cheap, broad scanning.
- **`fetch(url, max_chars)`** — Fetches and extracts a full web page via trafilatura. Expensive, selective deep reading.

The model learns *when* to search vs. fetch through reward signal alone.

### Training (GRPO)

[GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) generates multiple rollouts per question, scores them, and reinforces better trajectories:

1. Sample a question from the training set
2. Generate 4 rollouts — each is a multi-turn conversation where the model can call tools
3. Score each rollout with reward functions
4. Compute group-relative advantages (no critic network needed)
5. Update policy via clipped surrogate objective

We currently use [DAPO](https://arxiv.org/abs/2503.14476) loss (`loss_type="dapo"`) which removes GRPO's length bias. **Note:** SID-1 argues against this — they found length-debiased normalization causes OOV token collapse on long runs and keep per-sequence length normalization instead. This is a known divergence to revisit.

### Reward Functions

All rewards are in [0, 1], combined as a weighted sum:

| Reward | Weight | What it measures |
|--------|--------|-----------------|
| **Retrieval** | 1.0 | Does the ground-truth answer appear in any tool result (search snippets or fetched pages)? Binary match on significant words. |
| **Efficiency** | 0.5 | `max(0, 1 - extra_steps / gold_steps)`. Penalizes using more tool calls than the reference trajectory. |
| **Thinking** | 0.3 | Encourages brief `<think>` blocks (50 words = perfect, 200+ words = 0). No thinking = 0.5 (neutral). |
| **Truncation** | 0.3 | 1.0 if the model finishes naturally, 0.0 if truncated at token limit. |

### Data Generation

Training questions are generated synthetically using Claude Sonnet:

1. **Question generation** — Sonnet creates multi-hop questions (1-3 hops) requiring web search
2. **Trajectory generation** — Sonnet answers with search/fetch tools, full conversation recorded
3. **Quality judgment** — Sonnet evaluates trajectory quality, retries if needed
4. **Formatting** — Convert to TRL's conversational format with metadata (answer, hop count, gold tool count)

Distribution: 30% 1-hop, 40% 2-hop, 30% 3-hop. 200 training examples, 10 held-out eval.

## Training Configuration

```yaml
model_name: "Qwen/Qwen3-14B"
per_device_train_batch_size: 2
gradient_accumulation_steps: 4      # effective batch = 8
num_generations: 4                  # rollouts per question
max_completion_length: 1024
max_tool_calling_iterations: 3
loss_type: "dapo"
beta: 0.0                           # no KL penalty, no reference model
temperature: 0.7
learning_rate: 1e-5
num_train_epochs: 4
bf16: true
gradient_checkpointing: true

# LoRA (fits on single A6000)
lora_r: 16
lora_alpha: 32
lora_target_modules: "all-linear"
```

## Efficiency Improvements

Autoregressive generation dominates GRPO training time. Here's what we tried:

| Approach | Result | Notes |
|----------|--------|-------|
| Custom inference server (2 GPUs) | **2 min/step** | FastAPI on GPU 0, trainer on GPU 1. Weight sync via LoRA save + reload. |
| Single GPU baseline | 5 min/step | Sequential generate → train. |
| vLLM colocate | OOM | Model + KV cache exceeded 48GB. |
| vLLM server mode | Version conflict | `transformers<5` vs `>=5.2` — incompatible. |
| DDP (2 GPUs) | 15 min/step | Slower — no pipeline parallelism, double memory. |
| torch.compile | Never finished step 1 | Recompilation on every new sequence length kills autoregressive gen. |
| Static KV cache | Broke tool calling | Model stopped emitting `<tool_call>` tokens entirely. Likely incompatible with multi-turn insertion pattern. |
| Unsloth | Template conflict | Modifies `tokenizer.chat_template`, breaking TRL's prefix-preserving checks. |
| SDPA attention | **Free speedup** | Default in transformers 5.x, no downsides. |

## Current Results

### Training Metrics (250 steps)

Efficiency reward was broken for the first 200 steps (wrong message format — looked for Anthropic format, TRL uses OpenAI format). Fixed at step 200.

| Window | Reward | Retrieval | Efficiency | Truncation | Entropy |
|--------|--------|-----------|------------|------------|---------|
| Steps 100-119 | 0.709 | 0.306 | 0.000 | 0.844 | 0.964 |
| Steps 140-159 | 0.798 | 0.388 | 0.000 | 0.869 | 0.933 |
| Steps 200-209 | 1.078 | 0.312 | 0.750 | 0.800 | 0.517 |
| Steps 230+ | **1.346** | **0.542** | **0.833** | 0.792 | 0.140 |

Retrieval quality trending up. Efficiency reward providing signal after fix.

### Eval (Checkpoint 250, 10 held-out questions)

| Metric | Value |
|--------|-------|
| Answer correct | 10/10 (100%) |
| Used tools | 7/10 (70%) |
| Used fetch | 0/10 |
| Avg tool calls | 0.8 |

**Key finding:** The model is too smart for the eval set. It answers 3/10 questions from parametric knowledge without searching at all. Multi-hop questions collapse to single searches because the model already knows intermediate facts. The eval doesn't test what we care about — deliberate multi-step tool use.

## Open Issues & TODOs

### Reward Design

- **Partial credit for intermediate hops.** GRPO is a trajectory-level bandit — no per-step credit assignment. For multi-hop questions, 2 good hops + 1 bad = reward 0. This is the biggest reward design gap. Need a way to reward partial progress (gold sub-answers per hop, embedding similarity, or word overlap). Approach TBD.

- **Conditional rewards.** Efficiency and thinking only matter if retrieval succeeds. Consider gating them: `efficiency_final = efficiency * retrieval`. Don't reward efficient failures.

- **Drop truncation reward?** May condition the model to calibrate to the training token budget (1024). Retrieval reward already implicitly penalizes truncation. Possibly redundant.

- **Answer matching too strict.** Current check requires ALL significant words of the gold answer to appear in tool results. "Barack Obama" vs "Obama" = no match. Need fuzzy/partial matching.

- **Embedding-based snippet matching.** Compare model's retrieved content against gold trajectory snippets using embeddings. Solves "same content, different source" problem. Challenge: identifying which gold snippets are load-bearing.

### Eval & Data

- **Harder eval questions.** Need obscure facts, temporal questions (post-training-cutoff), and numerical lookups that genuinely require search. Current eval is solvable from parametric knowledge.

- **Populate answer_aliases.** Field exists in data but is empty. Would help with fuzzy answer matching.

- **Incentivize fetch().** Model never reads full pages — search snippets are sufficient for current questions. Need questions where snippets aren't enough.

### Alignment with SID-1

- **Switch to NDCG reward.** SID-1 uses NDCG over ranked document lists — partial credit with diminishing returns per rank. Our binary content match is much weaker signal.

- **Revert to length-biased GRPO.** SID-1 proves that length-debiased normalization (DAPO, what we use) causes gradient collapse toward OOV tokens on long runs. They keep per-sequence length normalization. We should test both and compare.

- **TI/TO token handling.** SID-1 processes all rollouts as raw token sequences, never converting to messages and back. TRL re-templates messages, which SID-1 found causes model collapse. Investigate whether TRL's approach introduces the same instability.

- **Length scheduling.** SID-1 starts with short max rollout length and gradually increases. We use fixed 1024. Implementing this could improve early training efficiency.

- **Format reward.** SID-1 added a format reward to prevent tool-calling format regression in later training stages. We don't have this yet and should monitor for format degradation.

### Infrastructure

- **Download checkpoints.** Checkpoints 50-250 are on stopped Vast.ai instance (contract 34089064). Need to download before host reclaims.

- **Static cache investigation.** `cache_implementation="static"` broke tool-call generation. Hypothesis: static KV cache can't handle multi-turn token insertion. Worth filing on huggingface/transformers.

### Future Optimization

- **[SPEC-RL](https://arxiv.org/abs/2509.23232)** — Reuse prior epoch rollouts as speculative drafts. 2.88x speedup, zero extra VRAM.
- **[SGLang](https://github.com/sgl-project/sglang)** — Already supports transformers 5.x. 29% throughput advantage over vLLM. RadixAttention helps with prefix-sharing in RL.
- **[veRL](https://github.com/volcengine/verl)** — Built for Search-R1, supports GRPO + multi-turn tool calling. Cleanest long-term migration path.

### Open-Source Contributions

- **TRL: Add SGLang backend** — `use_sglang=True` with colocate mode. No version conflicts.
- **Unsloth: Skip chat_template override** — Don't modify template when external trainer handles it.
- **vLLM: Help land transformers 5 support** — PR #30566, 130 commits awaiting review.

## Project Structure

```
src/
  env/            # SearchEnvironment (TRL protocol), providers, caching
  rewards/        # Retrieval, efficiency, thinking, truncation rewards
  training/       # GRPOTrainer, RemoteGRPOTrainer, inference server, configs
  data/           # Synthetic data generation pipeline
  eval/           # Baseline evaluation
  inference/      # Interactive tool-use inference
scripts/          # CLI entrypoints, cloud setup, monitoring
data/             # Training and eval JSONL datasets
logs/             # Training run logs
results/          # Evaluation results
checkpoints/      # LoRA adapter checkpoints
```

## Quick Start

```bash
# Install
pip install -e .

# Generate training data (requires ANTHROPIC_API_KEY)
python scripts/generate_data.py --count 200 --hops 1:3,2:4,3:3 --output data/generated.jsonl
python scripts/prep_dataset.py --input data/generated.jsonl --output data/train_trl.jsonl

# Train locally (single GPU)
python -m src.training.train --config src/training/configs/cloud_14b.yaml

# Train on 2x GPUs (pipeline mode)
bash scripts/train_dual_env.sh

# Test model with tools
python scripts/run_with_tools.py --model Qwen/Qwen3-14B
```

## References

- [SID-1 Technical Report](https://www.sid.ai/research/sid-1-technical-report) — Primary inspiration. Uses Magistral's modified GRPO with NDCG rewards, TI/TO token handling, and length scheduling.
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) — Base RL algorithm
- [DAPO](https://arxiv.org/abs/2503.14476) — GRPO length-bias removal (we use this, SID-1 argues against it)
- [Search-R1](https://arxiv.org/abs/2503.09516) — RL for search-augmented reasoning
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — RL-only training for reasoning
- [PipelineRL](https://github.com/ServiceNow/PipelineRL) — Async generation/training (OLMo-3)
- [MT-GRPO](https://arxiv.org/abs/2505.11821) — Turn-level credit assignment for multi-turn RL
- [TRL](https://github.com/huggingface/trl) — Training framework (v0.29.1)
