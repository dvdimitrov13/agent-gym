# Agent Gym: Building an RL-Trained Web Search Agent

## Technical Report — Full Development Journey

---

## 1. Project Goal

Train a language model (Qwen3-14B) to effectively search the web, read pages, and retrieve relevant information using **reinforcement learning only** — no supervised fine-tuning on tool-use demonstrations. Inspired by the [SID-1 Technical Report](https://www.sid.ai/research/sid-1-technical-report), which demonstrated that RL alone can teach models effective search behavior when rewarded for retrieval quality rather than supervised on search strategies.

The model learns two tools:
- **`search(query)`** — returns titles, URLs, and snippets from web search
- **`read(url, keywords)`** — fetches a page and returns the most relevant paragraphs via fuzzy keyword matching (rapidfuzz)

And a terminal action:
- **`submit_answer(passage_ids)`** — submits retrieved passages as the final answer, immediately ending the trajectory

---

## 2. Infrastructure & Compute

### Local Development
- Apple M1 Pro 16GB, conda `pytorch` environment
- Used for code development, data inspection, and LLM judge calls
- MPS backend for quick validation with Qwen3-0.6B

### Cloud Training
All training runs were conducted on [Vast.ai](https://vast.ai) GPU instances:

| Phase | GPU | VRAM | Cost/hr | Architecture |
|-------|-----|------|---------|-------------|
| V1 (Steps 1-600) | 2× RTX A6000 | 96GB total | ~$0.61 | Async dual-GPU: rollout server (GPU 0) + trainer (GPU 1) |
| V2 (Steps 600-1200) | Blackwell RTX PRO 6000 WS | 98GB | ~$0.83 | Single-GPU TI/TO |
| V3 (Steps 1-600) | Blackwell RTX PRO 6000 WS | 98GB | ~$0.82 | Single-GPU, DAPO + no thinking + curriculum |

### Framework Stack
- **TRL 1.0** (HuggingFace) — GRPOTrainer with `environment_factory` for tool-calling RL
- **Transformers 5.3** — model loading, generation
- **PEFT** — LoRA (r=16, alpha=32, all-linear targets)
- **PyTorch 2.7** with CUDA 12.8 (Blackwell SM100 architecture)

---

## 3. Data Pipeline

### V1: Sonnet-Generated Questions
We initially used Claude Sonnet to generate multi-hop factual questions with search trajectories. While the quality was excellent, the per-question cost was high due to Sonnet's pricing and the multi-step generation pipeline (question → trajectory → judge → auto-expand).

### V2: GPT 5.4 Cost Optimization
To reduce costs, we switched the generation pipeline to GPT 5.4. This introduced a critical data quality issue: GPT 5.4 frequently hallucinated dates, shifting 2024 events to 2025 (Nobel Prizes, Hurricane Helene, etc.). A DQ judge check caught 36/200 bad examples (18%).

We also discovered that multi-hop questions were being solved in a single hop during golden trajectory generation — the model already knew intermediate facts and took shortcuts. This meant the "golden" trajectories didn't actually demonstrate multi-hop search behavior.

### V3: Research-Grounded Regeneration
We fixed both issues by enabling web search during question generation. GPT 5.4 with web search access could:
1. Base questions on **actual current events** (2025-2026), avoiding hallucinated dates
2. Generate trajectories that **required genuine multi-hop search** because the model couldn't shortcut with parametric knowledge

All golden questions and trajectories were verified with an LLM-as-a-judge to ensure training and validation data quality. The judge evaluated whether answers were grounded in retrieved content and whether multi-hop chains were actually exercised.

**Final dataset:** 200 clean examples — 60× 1-hop, 80× 2-hop, 60× 3-hop, sorted by difficulty for curriculum learning.

**Eval dataset:** 22 questions about 2025-2026 events requiring genuine web search (not answerable from parametric knowledge).

---

## 4. Tool Design Evolution

### V1: `search()` + `fetch(url, page)`
Original design used page-turning: `fetch(url, page=1)`, `fetch(url, page=2)`, etc.

**Problem:** The model had to blindly turn pages hoping to find relevant content. Inefficient and hard to learn.

### V2: `search()` + `read(url, keywords)`
Replaced `fetch` with `read` — a Ctrl+F-like interface. The model passes keywords, and the tool returns the top-5 matching paragraphs via fuzzy matching (rapidfuzz).

**Key decision:** Search results include snippets so the model can answer directly from snippets when sufficient, and use `read()` only when it needs more detail. This teaches the model *when* to read, rather than forcing it.

### `submit_answer(passage_ids)`
Terminal action. Originally called `submit_ranking` (confusing — the model didn't understand what to submit). Renamed to `submit_answer` for discoverability.

**Critical behavior:** Immediately terminates the trajectory when called. No extra generation after submission.

---

## 5. Training Architecture

### Tokens-In / Tokens-Out (TI/TO)
From SID-1. Instead of decoding generated tokens to messages, re-applying the chat template, and re-tokenizing (lossy), we stay in **token space** throughout the multi-turn tool-calling loop.

The only decode/encode operations are:
1. Decode the tool call JSON (small, to extract function name + args)
2. Encode the tool result text (to splice into the token sequence)

The bulk of the conversation stays as original token IDs, avoiding the byte→token mismatches that SID-1 found cause training instability.

### V1: Async Dual-GPU (OLMo-3 Style)
Inspired by [OLMo-3](https://arxiv.org/abs/2512.13961) and [PipelineRL](https://arxiv.org/abs/2509.19128), the V1 architecture decouples generation from training:

- **GPU 0 (Actor):** Rollout server running TI/TO generation loops continuously. Generates rollouts into a buffer, serves completed rollouts to the trainer via HTTP. Computes behavior log-probabilities for off-policy correction.
- **GPU 1 (Learner):** Pulls completed rollouts from the actor, computes fresh logprobs under current policy for importance sampling, trains with GRPO loss, pushes updated LoRA weights to the actor after each step.

This asynchronous architecture means GPU 0 keeps generating while GPU 1 trains — no idle time waiting for the other.

### V2-V3: Single-GPU TI/TO
For the Blackwell RTX PRO 6000 (98GB), we consolidated to a single GPU. The large VRAM fits model + LoRA + gradient checkpointing comfortably, and eliminates HTTP overhead and weight sync complexity.

### TiToGRPOTrainer Implementation
Subclasses TRL's `GRPOTrainer`, overriding:
- **`_generate_single_turn()`** — injects ThinkingBudgetProcessor into model.generate()
- **`_tool_call_loop()`** — token-space tool dispatch with splice, snippet tracking
- **`_calculate_rewards()`** — multiplicative reward computation
- **`_batch_generate()`** — batched continuation generation with thinking prefix

Key design decisions:
- **No tool call = instant termination** — agent must always call tools
- **submit_answer = immediate stop** — no extra generation after submission
- **Only NEW tokens searched** — prevents re-detecting old spliced tool calls
- **Thinking tokens stripped between rounds** — prevents context blowup across iterations
- **Tool masks track splice boundaries** — `1` for model-generated, `0` for tool-result tokens

---

## 6. Training Runs

### V1: Steps 1-600 (Dual-GPU, Additive Rewards)

**Architecture:** Async dual-GPU with rollout server on GPU 0 and GRPO trainer on GPU 1.

**Reward:** Additive — `total = 1.0×judge + 0.5×efficiency + 0.5×format`

**Results:** Model learned to search and submit. At checkpoint 600, we evaluated on the 22-question eval set:
- **12/22 valid submissions** (55%), 7 direct answers without tools
- The low submit rate and tendency to answer from memory rather than search motivated our reward redesign and further training

### V2: Steps 600-1200 (Single-GPU, Multiplicative Rewards)

**Architecture change:** Moved to single Blackwell GPU (98GB). The large VRAM made dual-GPU unnecessary.

**Reward:** Switched to multiplicative — `reward = judge × format_scale × thinking_scale` (see Section 7 for why).

**Prompt engineering improvements:** Before continuing training, we improved the system prompt and eval tool-force settings. At checkpoint 600 with the improved prompt:
- **18/22 valid submissions** (~82%), up from 12/22
- **Judge average: 0.587**

This showed that prompt engineering alone could recover significant performance, but the model still needed reward signal improvement for further gains.

**Training continued to checkpoint 1200:**

| Model | Submit Rate | Judge Avg | vs Gold |
|-------|-----------|-----------|---------|
| Gold passages | 22/22 (100%) | 0.895 | 100% |
| Base Qwen3-14B | 9/22 (41%) | 0.291 | 33% |
| CP-600 (improved prompt) | 18/22 (82%) | 0.587 | 66% |
| CP-1200 | 20/22 (91%) | 0.655 | 73% |

CP-1200 achieved **2.25× improvement** over base model in judge score.

### V3: Steps 1-600 (DAPO, No Thinking, Curriculum)

**Key changes:**
- **DAPO loss** instead of GRPO (see Section 9)
- **Thinking disabled** — prepends empty `<think>\n</think>` block, forces immediate tool calls
- **Temperature annealing** — 1.0 (steps 0-200) → 0.7 (200-400) → 0.5 (400-600)
- **Curriculum:** 1-hop (steps 0-29, 384 tokens, 4 iters) → 2-hop (30-69, 512, 5) → 3-hop (70+, 768, 6)

**Training dynamics:**

| Phase | Steps | Avg Reward | Avg Grad Norm | Tool Calls/Step |
|-------|-------|-----------|---------------|-----------------|
| 1-hop | 1-30 | 0.631 | 0.45 | 2.7 |
| 2-hop | 31-70 | 0.696 | 0.55 | 3.0 |
| 3-hop | 71-600 | 0.716 | 1.5 | 4.1 |
| Final 20 | 580-600 | 0.714 | 2.8 | 4.1 |

Reward improved from 0.60 to 0.71 over training. Gradient norms grew as the model engaged more with harder questions. Tool calls per step nearly doubled (2.7 → 4.1) as the model learned to do multi-step retrieval.

**Eval results (CP-600):**

| Model | Submit Rate | Judge Score |
|-------|-----------|-------------|
| Base Qwen3-14B | 12/22 (55%) | 0.450 |
| V3 CP-600 | 13/22 (59%) | 0.485 |

V3 showed marginal improvement over base (+4% submit rate, +8% judge avg). Critically, **V3's low overall score is driven by format failures, not reasoning quality.** When the model successfully submitted, its judge score was 0.820 — essentially identical to the base model's 0.826 on submitted questions. The 9 failed questions all failed at the first generation step: the model emitted raw Python syntax (`search("query", 5)`) instead of the required `<tool_call>` XML format, causing the parser to reject the output and score it as zero.

**Head-to-head comparison:** On the 11 questions where both V2 CP-1200 and V3 CP-600 successfully submitted, retrieval quality is virtually identical:

| Model | Judge (11 overlapping questions) |
|-------|--------------------------------|
| V2 CP-1200 (with thinking) | 0.793 |
| V3 CP-600 (no thinking) | 0.798 |

The 8 questions where V2 submitted but V3 didn't span all difficulty levels (1-hop through 3-hop), and V2's own scores on those range from 0.23 to 1.00 — confirming that V3's failures are not correlated with question difficulty. The model simply falls into an invalid syntax path on those questions, regardless of how hard they are.

**Caveat:** The 11-question overlap skews easier:

| Hops | Overlap (both submitted) | Full eval set |
|------|-------------------------|---------------|
| 1-hop | 5 (45%) | 8 (36%) |
| 2-hop | 4 (36%) | 9 (41%) |
| 3-hop | 2 (18%) | 5 (23%) |

The parity in judge scores is encouraging, but a larger and more balanced eval set is needed to confirm that no-thinking retrieval quality truly matches thinking-enabled on harder multi-hop questions.

**Analysis:** The problem is mechanical, not cognitive. Without `<think>` blocks, the model can't internally deliberate on output format before committing tokens. It probabilistically falls into an alternative syntax path that the tool parser doesn't recognize. This is a cold-start format problem — the model was never shown tool-call demonstrations (pure RL), so it has no supervised prior on the correct XML structure. The RL training's `+15` logit boost for `<tool_call>` helped marginally (59% vs 55% submit rate) but wasn't enough to eliminate the alternative syntax path entirely.

This result is the strongest argument for investing in no-thinking training: the capability is already there — V3 matches V2's retrieval quality when it submits — only the format reliability needs to be solved. An SFT warmup phase to lock in the `<tool_call>` syntax before RL (see Section 13) would directly address this, potentially achieving V2-level quality at 4-5x lower latency.

### Training Comparison

The three runs are compared in the following interactive charts:

- **[Reward Trajectory](results/fig_reward_trajectory.html)** — V1 uses additive rewards (>1.0 scale), V2/V3 use multiplicative (0-1 scale). V2 and V3 converge to similar reward levels (~0.71) but through different mechanisms.
- **[Gradient Norms](results/fig_gradient_norms.html)** — V3 (DAPO) shows steadily increasing gradient norms as training progresses, unlike V1/V2 (GRPO) which are more stable.
- **[Tool Usage](results/fig_tool_usage.html)** — V3 ramps tool usage from 2.7 to 4.1 calls/step, showing the model learned to search more thoroughly over training.

---

## 7. Reward Design: The Full Journey

### Phase 1: Binary Answer Match
The simplest approach — check if the gold answer text appears in any tool result. This provided a binary signal (0 or 1) with no partial credit. It couldn't handle multi-hop questions where intermediate hops don't contain the final answer text.

### Phase 2: NDCG Over Ranked Passages (Abandoned)
Inspired by SID-1's use of NDCG over ranked document lists. We attempted embedding-based scoring using several approaches, each revealing a fundamental limitation:

**Finding 1 — Gold passage mismatch:** Comparing the model's live search results against gold passages from a different search session is fundamentally flawed. The same information appears with different wording across different URLs, and bi-encoder similarity (bge-small-en-v1.5) couldn't reliably detect semantic equivalence across these stylistic differences.

**Finding 2 — Threshold sensitivity:** Small embedding models give high baseline similarity for topically related content regardless of correctness. We could not find a threshold that reliably separated "right topic, right fact" from "right topic, wrong fact" — the similarity scores overlapped too much.

**Finding 3 — Multi-hop blind spot:** Both query-anchored scoring and cross-encoder scoring failed on multi-hop questions. An intermediate chain link like "Furukawa is president of Nintendo" scores low against the full question "What is the family name of the president who announced Switch 2?" The scoring has no concept of how intermediate hops contribute to the final answer.

**Finding 4 — No source authority:** Embedding similarity treats a Reddit comment and an official press release identically. There's no way to incentivize the model to prefer authoritative sources.

**Why we abandoned the approach:** The combination of session-dependent gold passages, threshold sensitivity, multi-hop blindness, and missing source authority made embedding-based NDCG unreliable as a training signal. We were building increasingly complex machinery (FAISS index, precomputed embeddings, sub-query decomposition) for a signal that still didn't work.

### Phase 3: LLM-as-a-Judge (RLAIF)
**GPT 4.1 mini, temperature=0.** A single API call per rollout replaces the entire NDCG pipeline. The judge sees the full trajectory (search queries, results, submitted passages) and scores:
- **Relevance** (50%): Do the submitted passages answer the question?
- **Completeness** (30%): Are all multi-hop chain links covered?
- **Source quality** (20%): Are sources authoritative and reliable?

**Cost:** ~$0.50 for an entire training run (~600 judge calls at 4 rollouts each).

**Key advantage:** The LLM judge naturally handles everything the embedding approach couldn't — it understands multi-hop chain relationships, evaluates source authority, and doesn't require gold passages from a specific search session. The judge evaluates quality holistically rather than through fragile numerical thresholds.

### Phase 4: Additive Combination (Steps 1-600)
`total = 1.0×judge + 0.5×efficiency + 0.5×format`

**Discovery:** The model learned to game the additive structure. It submitted garbage passage IDs (e.g., `[1, 2, 3]` instead of actual snippet IDs like `[S1, S2, S3]`) and still received full format reward (any submit_answer call scored 1.0). The additive structure allowed compensatory optimization — bad format but decent judge score still yielded acceptable total reward.

### Phase 5: Multiplicative Combination (Steps 600+)
`reward = judge_score × format_scale × thinking_scale`

- **format_scale** = valid_ids / total_ids (fraction of submitted IDs that exist in the snippet store). Returns 0 if no submission is made.
- **thinking_scale** = annealing from 1.0 to 0.0 over target+256 tokens (or always 1.0 when thinking is disabled)

**Why multiplicative works:** With multiplicative combination, every component must be satisfied — a zero in any component zeros the total reward. The model cannot compensate for submitting invalid passage IDs by having a good search trajectory. This matches findings from the GR³ paper (2026, arxiv:2603.10535), which demonstrated that additive reward combinations create "compensatory optimization shortcuts" in RL training.

We discovered this failure mode naturally before finding the paper — our training logs showed the model consistently submitting garbage IDs while maintaining decent total reward, which the multiplicative structure immediately eliminated.

---

## 8. Thinking Control

### The Problem
Qwen3 generates a `<think>...</think>` block before producing tool calls. Without control, the model spends its entire token budget on thinking and never actually calls tools.

### ThinkingBudgetProcessor
A custom LogitsProcessor that monitors tokens inside `<think>` blocks:
- **Soft nudge:** Boosts `</think>` logit by +2 to +15 as thinking approaches the token budget
- **After `</think>`:** Hard-forces `\n` (sets all other logits to `-inf`), then applies +15 soft boost for `<tool_call>`

**Evolution:** Initially implemented with a hard cap that forced `</think>` at the budget limit. We replaced this with reward annealing — the thinking_scale component of the multiplicative reward smoothly penalizes excessive thinking rather than cutting it off abruptly. This preserves exploration while still incentivizing concise reasoning.

### Disabling Thinking (V3)
For V3, we disabled thinking entirely by prepending `<think>\n</think>` to every generation. The model sees thinking as "already done" and goes straight to tool calls. The processor hard-forces `\n` after the closing tag, then soft-boosts `<tool_call>`.

**Hypothesis:** Removing thinking reduces token waste and forces the model to act immediately. SID-1 didn't use thinking blocks. The model should learn to go straight to tool calls.

---

## 9. Optimizations

### Training Speed

| Optimization | Result |
|-------------|--------|
| **Async dual-GPU (OLMo-3 style)** | GPU 0 generates rollouts while GPU 1 trains — no idle time. ~2 min/step vs ~5 min with sequential single-GPU |
| **Single Blackwell GPU (98GB)** | Fits everything in one GPU, eliminates HTTP overhead and weight sync |
| **SDPA attention** | Default in transformers 5.x, free speedup for attention computation |
| **Completion length curriculum** | Start at 384 tokens (1-hop), grow to 768 (3-hop). Shorter early steps train faster |
| **OOM-safe training step** | Wraps backward pass in try/except for `torch.cuda.OutOfMemoryError`, skips batch and clears cache instead of crashing |

### DAPO Loss (OLMo-3 Inspired)
Switched from standard GRPO to [DAPO](https://arxiv.org/abs/2503.14476) loss for V3, following OLMo-3's approach. DAPO applies four patches on top of GRPO:

1. **Token-level loss normalization** — GRPO divides by per-response length (biases toward short answers). DAPO normalizes across the full batch.
2. **Dynamic sampling** — if all rollouts in a group score identically (zero advantage variance), DAPO resamples. This addresses the zero-gradient problem we observed where the model produced near-identical rollouts.
3. **Clip higher** — widens the upper clip bound, allowing more aggressive exploration.
4. **Truncation masking** — exclude cut-off responses from loss. **Note:** We keep this OFF for TI/TO because multi-turn trajectories always hit max length.

### Temperature Annealing
V3 introduced temperature scheduling as a curriculum for exploration, inspired by the common practice in RL of starting with high exploration and gradually reducing it:

| Steps | Temperature | Rationale |
|-------|------------|-----------|
| 0-200 | 1.0 | High exploration — model discovers diverse search strategies |
| 200-400 | 0.7 | Moderate — consolidate learned patterns |
| 400-600 | 0.5 | Low — exploit best strategies, refine quality |

At temp=1.0, we observed more diverse rollouts and better DAPO advantage computation, addressing the zero-variance problem where all rollouts in a group received the same reward.

### Inference Optimizations (Eval)
- **Dual EOS suppression:** Qwen3 has two EOS tokens (`[151645, 151643]`). Both must be suppressed during generation to prevent premature termination inside thinking blocks.
- **Tool-force processor:** After `</think>`, hard-force newline then soft-boost `<tool_call>` by +15 logits. Ensures the model enters tool-calling mode reliably during evaluation.
- **Trafilatura timeout reduction:** Default 30s timeout with automatic retries caused 137s reads. Reduced to 10s.
- **Removed torch.compile:** Added 30s warmup with no benefit for variable-length generation. Eager mode is faster for autoregressive decoding.

---

## 10. Curriculum Learning

### Data Sorting
Training data sorted by difficulty: 1-hop questions first, then 2-hop, then 3-hop. The model learns simple search→submit patterns before tackling multi-step retrieval.

### Progressive Schedules

| Schedule | Steps 0-29 | Steps 30-69 | Steps 70+ |
|----------|-----------|-------------|-----------|
| Max completion tokens | 384 | 512 | 768 |
| Max tool iterations | 4 | 5 | 6 |

### Reward Trajectory by Phase (V3)

| Phase | Steps | Avg Reward |
|-------|-------|-----------|
| 1-hop curriculum | 1-30 | 0.617 |
| 2-hop curriculum | 31-70 | 0.721 |
| 3-hop curriculum | 71-150 | 0.670 |

The jump from 1-hop (0.617) to 2-hop (0.721) shows the model quickly learned the search→submit pattern. The 3-hop drop to 0.670 reflects harder questions, not regression.

---

## 11. Evaluation

### Eval Setup
- 22 questions about 2025-2026 events requiring genuine web search
- LLM judge (GPT 4.1 mini) scores relevance (50%), completeness (30%), source quality (20%)
- Same judge used for training and evaluation
- `--disable-thinking` flag matches training setting for V3

### Results Across All Runs

| Model | Submit Rate | Judge Score | vs Gold |
|-------|-----------|-------------|---------|
| Gold passages | 22/22 (100%) | 0.895 | 100% |
| Base Qwen3-14B | 12/22 (55%) | 0.450 | 50% |
| V1 CP-600 (additive) | 12/22 (55%) | 0.450 | 50% |
| V1 CP-600 (improved prompt) | 18/22 (82%) | 0.587 | 66% |
| **V2 CP-1200 (multiplicative)** | **20/22 (91%)** | **0.655** | **73%** |
| V3 CP-600 (DAPO, no think) | 13/22 (59%) | 0.485 | 54% |

### Key Findings

1. **V2 CP-1200 is the best model overall** — 91% submit rate with 0.655 judge average. The combination of thinking + multiplicative rewards + 1200 steps produced the most reliable tool-using agent.

2. **Disabling thinking hurt submit rate, but not retrieval quality** — V3 only submits on 59% of questions vs V2's 91%, but when it does submit, quality is nearly identical (0.820 vs V2's judge scores). The gap is driven by format failures — the model emits invalid tool-call syntax on 41% of questions — not by degraded reasoning.

3. **Prompt engineering matters significantly** — V1 CP-600 jumped from 55% to 82% submit rate just by improving the system prompt, without any additional training.

4. **Base model is surprisingly capable when it submits** — 0.826 judge average on submitted questions. The base model already knows how to search well; the RL training primarily teaches it to *consistently* use tools rather than answering from memory.

### Latency Comparison (Blackwell GPU)

| Iterations | V2 (with thinking) | V3 (no thinking) | Speedup |
|-----------|-------------------|------------------|---------|
| 1 | 8.0s | 0.7s | 11x |
| 2 | 15.0s | 3.3s | 4.5x |
| 3 | 25.5s | 6.0s | 4.3x |

Disabling thinking yields a **4-5x latency improvement** on tool-using queries (2-3 iterations). The thinking tokens dominate generation time in V2 — the model spends most of its token budget reasoning before acting. V3 skips this entirely, going straight to tool calls.

This latency advantage is a strong motivation for further work on no-thinking training. Since V3's low scores are driven by format failures (41% invalid tool-call syntax) rather than degraded reasoning — submitted questions score 0.820, matching the base model — the path to production-quality no-thinking inference is solving a mechanical format problem, not a fundamental capability gap. An SFT warmup phase on tool-call format demonstrations combined with a format reward floor (see Section 13) would directly address the root cause.

---

## 12. Lessons Learned

### 1. Reward design is everything
We spent more time on reward design than any other component. Binary matching → NDCG with embeddings → LLM judge → additive combination → multiplicative combination. Each iteration fixed a specific gaming behavior. The multiplicative structure was the breakthrough — it eliminated compensatory shortcuts entirely.

### 2. Data quality is the #1 priority
18% of GPT-generated questions had hallucinated dates. Bad data poisons RL — the model learns to ramble about contradictions instead of searching. The web-search-enabled generation fix was simple but critical.

### 3. Start simple, add complexity only when needed
Our most productive setup was the simplest: single GPU, TI/TO trainer, LLM judge. FAISS embeddings, NDCG scoring, and cross-encoder scoring all added complexity without proportional benefit.

### 4. Multiplicative > additive for multi-component rewards
Additive rewards let the model game one component to compensate for another. With multiplicative, every component must be satisfied. We discovered this naturally and later found backing in the GR³ paper.

### 5. Always verify learning is happening
Check `grad_norm` and `entropy` early. If both are zero, the model isn't learning regardless of what rewards look like. We ran for 140+ steps with zero gradients before noticing, due to `mask_truncated_completions` being incompatible with our multi-turn setup.

### 6. model.eval() during generation is non-negotiable
LoRA dropout in training mode corrupts autoregressive generation. This caused garbage output (Chinese characters, repeated tokens) in every round-2 generation until discovered. Always wrap `model.generate()` with eval mode.

### 7. Prompt engineering matters even in RL
Improved system prompts and tool descriptions at CP-600 raised submit rate from 55% to 82% without any additional training. RL learns the policy, but the prompt frames what's possible.

---

## 13. Future Research Directions

### Partial Credit for Intermediate Hops
GRPO is a trajectory-level bandit — no per-step credit assignment, no critic, no V(s). For multi-hop questions, intermediate hops build on each other, but the final reward only reflects the end result. Two good hops + one bad hop = zero reward, wasting useful signal from the successful hops. With only 4 rollouts per prompt, GRPO can't statistically isolate which hop made the difference. Approaches to explore: gold sub-answer matching, turn-level credit assignment ([MT-GRPO](https://arxiv.org/abs/2505.11821)), or learned value functions.

### Speculative Decoding for RL (SPEC-RL)
[SPEC-RL](https://arxiv.org/abs/2509.23232) reuses previous epoch's rollouts as implicit drafts for speculative decoding, achieving **2.88× speedup** on Qwen-3-8B with DAPO — with zero extra VRAM (no draft model needed). This is the most promising generation speedup for our setup since it works without architectural changes.

### SGLang as Inference Backend
[SGLang](https://github.com/sgl-project/sglang) already supports transformers 5.x (unlike vLLM), offers 29% throughput advantage over vLLM, and provides RadixAttention for up to 6.4× improvement on prefix-sharing (common in RL rollouts). TRL doesn't support SGLang natively yet — this is a potential open-source contribution.

### veRL Migration
[veRL](https://github.com/volcengine/verl) (used by Search-R1) natively supports GRPO with multi-turn tool calling via SGLang. It would provide a cleaner long-term architecture than our TRL-based approach, eliminating the version conflicts and custom overrides we maintain.

### Conditional Reward Gating
Multiply efficiency and thinking rewards by the judge score: `efficiency_final = efficiency × judge`. Rationale: process quality (efficient steps, good thinking) only matters if the outcome is good — don't reward efficient failures. Our training logs showed 29% of steps had high efficiency but low retrieval quality, meaning we were rewarding the wrong behavior.

### Off-Policy Training for Hardware Utilization
Our V1 async dual-GPU architecture (OLMo-3 style) demonstrated the viability of decoupling generation from training. Further exploration should focus on: scaling to more than 2 GPUs (multiple actors feeding a single learner), managing rollout staleness as the policy diverges from the behavior policy, and implementing in-flight weight updates (OLMo-3's key innovation where actors receive new weights mid-generation without invalidating the KV cache). This would enable efficient utilization of heterogeneous GPU setups — cheap inference GPUs generating rollouts while expensive training GPUs focus solely on gradient updates.

A particularly promising direction is **OAPL** (Optimal Advantage-based Policy Optimization with Lagged Inference policy, Ritter et al., 2026), which formalizes large-batch iterative off-policy RL. OAPL explicitly addresses the staleness problem by combining A*PO-style advantage estimation with KL regularization against a reference policy, preventing the training policy from diverging too far from the lagged inference policy used for rollout generation. Unlike our current β=0 setup (no KL penalty), OAPL's KL-regularized objective provides principled stability guarantees as the lag between generation and training increases — a critical property when scaling to multiple asynchronous actors.

### SFT Warmup for No-Thinking Training

V3's format failure analysis (Section 6) reveals a cold-start problem: without thinking, the model probabilistically falls into invalid tool-call syntax because pure RL never demonstrated the correct format. A short SFT warmup phase before RL could solve this directly.

[R1-Searcher++](https://arxiv.org/abs/2505.17005) validates this approach — it uses an SFT cold-start for "preliminary format learning" followed by RL for dynamic search strategy discovery. The key insight: SFT teaches the mechanical `<tool_call>` XML syntax and `submit_answer` structure (a narrow, well-defined skill), while RL teaches *when* and *what* to search (the open-ended strategic behavior). These are complementary, not competing.

The warmup should be minimal — ~50-100 steps on format demonstrations — to avoid narrowing exploration during the subsequent RL phase. The demonstrations need only show correct tool-call syntax, not optimal search strategies. Combined with a format reward floor (below), this would address both the cold-start problem (SFT teaches format) and the gradient signal problem (floor maintains learning pressure on format-invalid trajectories).

### Decomposed Multi-Judge Reward

Our current reward uses a single LLM judge call that scores relevance, completeness, and source quality simultaneously. This holistic approach is simple but produces noisy, entangled scores — the judge must weigh multiple criteria in one pass, and the model receives a single blended signal that doesn't indicate *which* aspect to improve.

A more principled design decomposes the reward into three parallel LLM judge calls, each evaluating one orthogonal criterion:

1. **Retrieval NDCG** — The judge sees the question and all retrieved snippets, then selects only passages that truly answer the question as the ideal submission. NDCG between the model's actual `submit_answer` and this ideal set scores submission quality. If the model searched poorly and found nothing relevant, the ideal set is empty and any submission scores 0 — so search quality is implicitly captured. If the model found good passages but submitted the wrong IDs, NDCG is low.

2. **Trajectory efficiency** — The judge sees the question and full trajectory (queries issued, pages read, iterations used). Scores how direct the search strategy was — redundant queries, unnecessary reads, and wasted iterations lower the score.

3. **Source quality** — The judge sees the submitted passages and their source URLs. Scores authoritativeness (official sites, major news outlets score higher than blogs/forums).

All three calls fire in parallel, so latency matches the current single-call setup. Cost triples (~$0.003/rollout) but remains negligible. The final reward stays multiplicative:

```
reward = NDCG(ideal, actual) × efficiency_scale × source_quality_scale
```

This design is backed by **Rubrics as Rewards (RaR)** ([arxiv 2507.17746](https://arxiv.org/abs/2507.17746)), which found that decomposing LLM judge evaluation into separate criteria-specific calls achieves **31% improvement on HealthBench** over holistic single-call judging. RaR also found that decomposed rewards **reduce judge variance**, particularly with smaller judge models — the structured per-criterion prompts act as anchors that prevent noisy holistic scoring. The NDCG component is separately validated by **Rec-R1** ([arxiv 2503.24289](https://arxiv.org/abs/2503.24289)), which uses NDCG@K directly as a GRPO reward signal and found it stabilizes convergence compared to sparser metrics.

### Vanishing Gradient Floor
When format_scale=0 (model emits invalid tool-call syntax), the entire reward is zeroed — the judge score gets no gradient, so the model can't learn *what* to improve about its retrieval strategy. This is especially damaging for no-thinking training, where 41% of V3 trajectories hit this zero.

Adding a floor `max(format_scale, 0.01)` would maintain a small gradient signal even on format-invalid submissions. The [Progressive Reward Shaping (PRS)](https://arxiv.org/abs/2512.07478) paper addresses this exact problem in GRPO — binary format rewards cause gradient death when all samples in a group fail format (advantage = 0, no learning). PRS suggests graduated partial credit (e.g., has tool call but wrong structure = 0.3, completely wrong = 0.01, correct format = 1.0) for richer gradient signal. Even the simple floor approach would prevent complete gradient death on the 41% of V3 trajectories that currently contribute nothing to learning.

### Open-Source Contributions
Several compatibility issues we encountered represent contribution opportunities:
- **TRL + SGLang backend** — would benefit the entire RL-for-LLM community
- **Static KV cache + tool calling investigation** — `cache_implementation="static"` breaks multi-turn tool-call generation, likely a transformers bug worth filing
- **PipelineRL backend abstraction** — currently hardcoded to vLLM, could support SGLang or custom backends

---

## Appendix A: Paper References

| Paper | Relevance |
|-------|-----------|
| [SID-1 Technical Report](https://www.sid.ai/research/sid-1-technical-report) | Primary inspiration — RL for web search, TI/TO approach |
| [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) | Base RL algorithm |
| [DAPO](https://arxiv.org/abs/2503.14476) | GRPO stability fixes — token-level loss, dynamic sampling |
| [GR³](https://arxiv.org/abs/2603.10535) | Multiplicative reward design — why additive fails |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Closest prior work — RL for search-augmented reasoning |
| [MT-GRPO](https://arxiv.org/abs/2505.11821) | Turn-level credit assignment for multi-turn |
| [OLMo-3](https://arxiv.org/abs/2512.13961) | Off-policy GRPO with PipelineRL, DAPO loss |
| [PipelineRL](https://arxiv.org/abs/2509.19128) | Async actor-learner architecture |
| [SPEC-RL](https://arxiv.org/abs/2509.23232) | Speculative decoding for RL — 2.88× speedup |
| [R1-Searcher++](https://arxiv.org/abs/2505.17005) | SFT cold-start + RL for search — format warmup |
| [PRS](https://arxiv.org/abs/2512.07478) | Progressive reward shaping — gradient floor for format failures |
| [RaR](https://arxiv.org/abs/2507.17746) | Decomposed multi-criteria LLM judge rewards — 31% gain over holistic |
| [Rec-R1](https://arxiv.org/abs/2503.24289) | NDCG@K as GRPO reward signal — stabilizes convergence |

## Appendix B: Cost Summary

| Item | Estimated Cost |
|------|---------------|
| V1 training (600 steps, 2×A6000) | ~$12 |
| V2 training (600 steps, Blackwell) | ~$15 |
| V3 training (600 steps, Blackwell) | ~$12 (in progress) |
| LLM judge calls (all runs) | ~$2 |
| Idle compute (debugging, waiting) | ~$10 |
| **Total** | **~$51** |
