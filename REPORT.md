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

### V3: Steps 1-600 (DAPO, No Thinking, Curriculum) — In Progress

**Key changes:**
- **DAPO loss** instead of GRPO (see Section 10)
- **Thinking disabled** — prepends empty `<think>\n</think>` block, forces immediate tool calls
- **Temperature annealing** — 1.0 (steps 0-200) → 0.7 (200-400) → 0.5 (400-600)
- **Curriculum:** 1-hop (steps 0-29, 384 tokens, 4 iters) → 2-hop (30-69, 512, 5) → 3-hop (70+, 768, 6)

**Status at step 185/600:** avg reward 0.603 → 0.663 (improving), healthy gradients (0.97-1.63).

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

*[Section reserved for final eval results after V3 training completes.]*

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

### Vanishing Gradient Floor
When format_scale=0 (model submits invalid IDs), the judge reward gets zero gradient — the model can't learn *what* to improve. Adding a floor `max(format_scale, 0.01)` would maintain a small gradient signal even on fully invalid submissions. Recommended but not yet implemented.

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

## Appendix B: Cost Summary

| Item | Estimated Cost |
|------|---------------|
| V1 training (600 steps, 2×A6000) | ~$12 |
| V2 training (600 steps, Blackwell) | ~$15 |
| V3 training (600 steps, Blackwell) | ~$12 (in progress) |
| LLM judge calls (all runs) | ~$2 |
| Idle compute (debugging, waiting) | ~$10 |
| **Total** | **~$51** |
