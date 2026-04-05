# NEXUS-AURORA: Autonomous Unified Reasoning Architecture
## Design Specification — Project NEXUS

**Date:** 2026-04-05
**Version:** 2.0 (Replaces HRD spec — new first-principles architecture)
**Status:** Approved for implementation planning
**Objective:** Build a genuinely novel ~46M-parameter LLM trained from scratch on Kaggle T4×2 that structurally separates cognitive functions — surface generation, private reasoning, and self-verification — into distinct architectural streams, addressing the five fundamental limitations of current transformer-based LLMs.

---

## 1. The Five Fundamental Limitations (First-Principles Analysis)

These limitations were derived by introspecting on what current LLMs — including Claude — cannot do architecturally, not by observing benchmark failures:

| # | Limitation | Root Cause |
|---|-----------|-----------|
| L1 | Cannot think without producing visible tokens | Single residual stream: all computation leads directly to token output |
| L2 | Uniform compute per token regardless of difficulty | No mechanism to signal "this position needs more thought" |
| L3 | All layers perform identical operations | No structural specialization — syntax/semantics/logic all share same layer design |
| L4 | Cannot revise or flag errors in previous representations | Residual stream flows forward only; no internal backward signal |
| L5 | Attention is relational, not causal | Attention weights express correlation, never directed causation or implication |

**Existing work and why it doesn't solve these:**

| Paper | What it does | Limitations not addressed |
|-------|-------------|--------------------------|
| Think-at-Hard (2025) | Selective iterations via LoRA adapters, same stream | Fine-tuning only, same stream, no private workspace |
| Inner Thinking Transformer, ACL 2025 | Adaptive depth routing | Single stream, no private reasoning |
| Dual-Stream Transformer (2026) | Attention stream + FFN stream | For interpretability only, no adaptive compute, no reasoning |
| Looped Transformers (Ouro) | Recursive block reuse | All tokens loop uniformly, no per-position adaptivity |
| Titans | Neural long-term memory | For context length, not per-position private reasoning |

**The confirmed gap:** No existing architecture trained from scratch combines (1) a dedicated private reasoning stream that never produces tokens, (2) adaptive iteration count per position, (3) typed layers for different cognitive functions, (4) a verification gate with auxiliary backward signal, and (5) sparse causal bridging between abstractions and evidence.

---

## 2. Core Architecture: NEXUS-AURORA

### 2.1 The Three-Stream Design

AURORA replaces the transformer's single residual bus with three functionally distinct streams trained jointly from scratch:

```
Input Tokens
      │
      ▼
┌─────────────────────────────────────┐
│  SURFACE STREAM  (S)                │
│  d_s = 512  │  Causal attention     │  → produces output tokens
│  6 blocks   │  Standard generation  │
└──────────────────┬──────────────────┘
                   │ Bridge Layer (cross-attention S → R, R → S)
                   ▼
┌─────────────────────────────────────┐
│  REASONING STREAM  (R)              │
│  d_r = 256  │  32 reasoning slots   │  → NEVER produces tokens (private)
│  6 blocks   │  K = {1,2,4} iters   │  → K driven by difficulty(S)
│             │  Sparse causal bridge │
└──────────────────┬──────────────────┘
                   │ Verify Gate (cross-attention S ↔ V)
                   ▼
┌─────────────────────────────────────┐
│  VERIFICATION STREAM  (V)           │
│  d_v = 64   │  Surprise signal      │  → auxiliary training loss only
│  per-position│  confidence score    │  → gates R→S write-back
└──────────────────┬──────────────────┘
                   │
                   ▼
              Output Logits
             (from S only)
```

**Key principle:** S is the only stream that produces tokens. R and V exist purely to improve the quality of S's representations. This is architecturally guaranteed — R and V have no output projection to vocabulary.

### 2.2 Addressing Each Limitation

**L1 — Private computation:**
The Reasoning Stream R has its own parameters, its own attention, its own representations. It processes in parallel with S but never generates tokens. All of R's computation is invisible to the output — it exists only to build richer representations that flow back into S via the Bridge Layer. This is not chain-of-thought; no text is produced.

**L2 — Adaptive compute per position:**
After each block's Surface Layer processes position `i`, a lightweight Difficulty Estimator (MLP) scores the position's hidden state and outputs K ∈ {1, 2, 4}:

```
logits_i = W₂ · GELU(W₁ · s_i)       # shape: (3,) for K∈{1,2,4}
K_i = argmax(softmax(logits_i)) → {1,2,4}
```

**Batching policy (GPU efficiency):** Within each block at training time, K is set to `max(K_i)` across all positions in the batch. All positions run the same number of R iterations — but the Difficulty Estimator still learns meaningful gradients because the K logits affect the loss through Gumbel-softmax straight-through. At inference time, sequences can be padded and batched by K value for true per-position adaptive compute.

Easy tokens (punctuation, function words) converge toward K=1. Hard tokens (logical connectives, mathematical operations, rare concepts) converge toward K=4. The K distribution histogram is a primary diagnostic during training.

**L3 — Layer specialization:**
Each block contains 4 typed sublayers in fixed order. Different types do structurally different things:

```
Block = Pattern Layer → Semantic Layer → Bridge Layer → Verify Layer
```

Details in Section 2.3.

**L4 — Internal backward signal:**
The Verification Stream V reads both S and R after each block and computes a surprise score — how inconsistent is the current surface token's representation with what the reasoning stream "expected"? This surprise is used as an auxiliary training loss:

```
L_surprise = mean(surprise_score_i) across all positions
L_total = L_CE + λ_v · L_surprise    (λ_v = 0.1)
```

At inference time, V's surprise score gates how strongly R writes back to S — high surprise → stronger write-back (R overrides S's representation), low surprise → weak write-back (S is already consistent with R).

**L5 — Implicit causal structure:**
The Bridge Layer uses sparse top-k attention from R to S: each of the 32 reasoning slots attends to at most k=8 surface token positions (top-k by relevance score). This creates a sparse directed graph — reasoning abstractions point to specific evidence positions, not all positions equally. Over training, R learns to build concept representations that are grounded in specific evidence, not distributed attention soup.

---

## 3. Layer Types — Typed Architecture

### 3.1 Pattern Layer (P) — Local Syntax and N-gram Patterns

```
Attention: Causal local window attention, window_size = 64 (look-back only, never forward)
FFN: SwiGLU, d_ffn = 512 (1× ratio, narrow — local patterns don't need width)
Normalization: RMSNorm (Pre-LN placement)
Purpose: Captures syntax, morphology, local collocations
```

This layer sees only the 64 nearest preceding tokens (causal — no future leakage). It cannot access long-range context. This constraint forces the Pattern Layer to specialize in local structure. In standard transformers, this specialization emerges accidentally — here it's architecturally enforced.

### 3.2 Semantic Layer (Se) — Long-Range Meaning Integration

```
Attention: Full causal attention (GQA, 8 query heads, 2 KV heads)
FFN: SwiGLU, d_ffn = 2048 (4× ratio, wide — semantic integration needs capacity)
Normalization: RMSNorm (Pre-LN placement)
Purpose: Long-range dependencies, semantic coherence, discourse structure
```

GQA (Grouped Query Attention) reduces KV cache at inference — important for deployment. The wide FFN here is the main "fact storage" component.

### 3.3 Bridge Layer (Br) — Surface ↔ Reasoning Cross-Communication

```
S → R attention: Surface positions query Reasoning slots (S reads from R)
R → S attention: Reasoning slots query Surface positions, top-k=8 sparse (R writes to S)
Normalization: RMSNorm on both streams
Gating: V's surprise score gates R→S write-back strength
Purpose: Bidirectional information transfer between streams
```

Mathematical formulation:
```
# S reads from R (dense — surface should see all reasoning)
s_new = s + Attention(Q=s, K=r, V=r)

# R writes to S (sparse — reasoning grounds itself in specific evidence)
attn_scores = Q_r · K_s^T / √d
sparse_mask = top_k(attn_scores, k=8)  # zero out non-top-k
r_to_s = sparse_attn(Q=s, K=r_selected, V=r_selected)
s_final = s + gate_v · r_to_s          # gate_v from Verification Stream
```

### 3.4 Verify Layer (Ve) — Consistency Check

```
Attention: Cross-attention between S and V (tiny d_v=64)
Purpose: Compute surprise signal, gate Bridge write-back
Output: per-position surprise score ∈ [0,1]
```

Mathematical formulation:
```
surprise_i = sigmoid(W_v · [s_i ; r_i])   # concatenate surface + reasoning
gate_v_i = 1 - surprise_i                 # high surprise → strong R override
L_surprise = mean(surprise_i)             # auxiliary loss
```

The surprise score has a dual role:
- **Training:** Penalizes inconsistency between S and R (auxiliary loss)
- **Inference:** Dynamically gates how strongly R corrects S

---

## 4. Reasoning Stream (R) — Internal Architecture

R is a smaller transformer running in parallel with S:

```
d_r = 256
n_reasoning_slots = 32   (these are like "concept slots", not tied to token positions)
n_layers_R = 6           (one per block, same depth as S)
attention_R: Full attention among the 32 slots (not causal — R is not autoregressive)
FFN_R: SwiGLU, d_ffn_r = 512 (2× ratio)
```

**Key design choice: R is NOT autoregressive internally, but is causally grounded via S.**

R's 32 slots attend to each other without causal masking (full self-attention among slots). This allows reasoning to be holistic — a concept slot can be influenced by all other concept slots simultaneously.

**Critical: No future leakage.** When R reads from S (via Bridge Layer), it uses causally-masked cross-attention — R slot `r_j` can only attend to S positions ≤ current position. R's internal self-attention is non-causal only among the 32 slots (which represent concepts, not positions). This distinction prevents future token information from leaking into past positions through the R→S write-back.

**R initialization:** The 32 reasoning slots are initialized as learned parameters (not from token embeddings). They start as random vectors and learn to represent abstract reasoning primitives over training.

**R update within a block:**
```
For each block b:
  1. r = SelfAttention_R(r)          # reason among slots
  2. r = FFN_R(r)                    # transform slots
  3. [repeat K_i times for position i]
  4. r participates in Bridge Layer  # communicate with S
```

---

## 5. Positional Encoding — CoPE on Surface Stream Only

CoPE (Contextual Positional Encoding) is applied to S only. R uses no positional encoding — the 32 reasoning slots are positionally unordered (they represent concepts, not positions). This is intentional: reasoning should be position-agnostic.

```
# CoPE on S:
gate_i = softmax(q_s · e_i)            # 16 context vectors
pos_s = Σ gate_i · position_embed_i    # contextual position
```

The Reasoning Stream R receives positional information implicitly through the Bridge Layer — it learns which surface positions to ground each reasoning slot in, via sparse top-k attention.

---

## 6. Full Block Execution Order

```
Input: s (B, T, 512), r (B, 32, 256), v_score (B, T, 1)

Step 1 — PATTERN LAYER (on S):
  s = s + LocalAttention(s, window=64)
  s = s + SwiGLU_narrow(s)

Step 2 — SEMANTIC LAYER (on S):
  s = s + GQA(s, CoPE_positions)
  s = s + SwiGLU_wide(s)

Step 3 — DIFFICULTY ESTIMATION (per position, Gumbel-softmax for gradients):
  K_logits = MLP(s)                           # (B, T, 3) — logits for K∈{1,2,4}
  K_soft = gumbel_softmax(K_logits, tau=1.0)  # differentiable during training
  K_hard = argmax(K_logits) → {1,2,4}         # actual discrete K used
  K_batch = max(K_hard) across sequence        # GPU efficiency: run max K for all

Step 4 — REASONING ITERATIONS (on R, K times):
  for _ in range(K):
    r = r + SelfAttention_R(r)               # reason among slots
    r = r + FFN_R(r)

Step 5 — BRIDGE LAYER (S ↔ R):
  s = s + Attention(Q=s, K=r, V=r)                    # S reads R (dense, all 32 slots)
  r_from_s = causal_cross_attn(Q=r, K=s, V=s)        # R grounds in S (CAUSAL — no future leak)
  r = r + r_from_s                                    # update R slots with grounded info
  r_sparse = sparse_top_k_attention(Q=s, K=r, V=r, k=8)  # S gets focused reasoning signal
  s = s + gate_v * r_sparse                           # gated write-back from R to S

Step 6 — VERIFY LAYER (S ↔ V):
  surprise = sigmoid(W_v · concat(s, r))
  gate_v = 1 - surprise                       # update gate for next block
  L_surprise += mean(surprise)                # accumulate aux loss

Output: s (updated), r (updated), gate_v (updated), L_surprise (accumulated)
```

6 blocks total. Each block adds: L_surprise_b → summed → L_total = L_CE + 0.1 × mean(L_surprise).

---

## 7. Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| Surface embedding (8192 × 512) | 4.2M | Tied with output projection |
| S: Pattern Layers × 6 | 11.0M | Causal local attn (4×512²) + narrow SwiGLU (3×512²) |
| S: Semantic Layers × 6 | 22.8M | GQA (8q,2kv) + wide SwiGLU (3×512×2048) |
| R: Internal layers × 6 | 3.9M | Self-attn (4×256²) + SwiGLU (3×256×512), d_r=256 |
| Bridge Layers × 6 | 7.8M | S↔R cross-attn both directions |
| Verify Layers × 6 | 0.3M | Linear (512+256)→64→1 per position |
| Difficulty Estimators × 6 | 0.4M | MLP 512→128→3 (logits for K∈{1,2,4}) |
| CoPE position vectors | 0.008M | 16 vectors × 512 |
| **Total** | **~50M** | |

---

## 8. Training Configuration

### 8.1 Dataset

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| FineWeb-Edu (subset) | ~1.5B | Main training — high quality, curated |
| TinyStories | ~500M | Ablation sanity checks |
| Validation | 1% of FineWeb-Edu | BPB tracking |

Pre-tokenized: `train.bin`, `val.bin` (np.memmap, uint16, BPE vocab=8192 via SentencePiece).

### 8.2 Hyperparameters

```yaml
model:
  d_surface: 512
  d_reasoning: 256
  d_verify: 64
  n_reasoning_slots: 32
  n_blocks: 6
  n_heads_S: 8              # GQA: 8 query, 2 KV heads
  n_heads_R: 4              # full attention among 32 slots
  d_ffn_pattern: 512        # 1× ratio (narrow)
  d_ffn_semantic: 2048      # 4× ratio (wide)
  d_ffn_R: 512              # 2× ratio
  local_window: 64          # Pattern Layer window
  bridge_top_k: 8           # sparse R→S attention
  cope_positions: 16
  vocab_size: 8192
  max_seq_len: 512

training:
  batch_size: 16             # smaller — AURORA is more memory-intensive
  gradient_accumulation: 8   # effective batch = 128
  max_lr: 1e-3
  min_lr: 1e-4
  warmup_tokens: 100_000_000
  stable_tokens: 1_300_000_000
  decay_tokens: 100_000_000
  grad_clip: 1.0
  weight_decay: 0.1
  dtype: float16
  surprise_loss_weight: 0.1  # λ_v

optimizer:
  name: muon                 # 2× compute efficiency vs AdamW at small scale
  momentum: 0.95
  nesterov: true
  embed_lr: 3e-4             # embeddings use AdamW
  verify_lr: 1e-3            # verification stream needs higher LR

init:
  scheme: u-muP              # hyperparameter transfer to larger scales
  reasoning_slots: learned   # initialized as random unit vectors
  difficulty_bias: -1.0      # bias toward K=1 initially (let it learn to think harder)

hardware:
  gpus: 2
  strategy: ddp
  gradient_checkpointing: true   # AURORA needs this — 3 streams in memory
```

### 8.3 Optimizer — Muon

Muon (Newton-Schulz orthogonalization) is applied to all matrix parameters across all three streams. AdamW for embeddings and the tiny difficulty estimator. Muon achieves ~2× compute efficiency vs AdamW at small scale with better convergence.

### 8.4 Schedule — WSD (Warmup-Stable-Decay)

Mathematically proven optimal over cosine decay (ICML 2025). Three phases:
- Warmup 100M tokens: streams learn basic roles
- Stable 1.3B tokens: core training, all three streams specialize
- Decay 100M tokens: final compression

### 8.5 Memory Management (Critical for T4)

AURORA has three streams in memory simultaneously. FP16 estimates:
- S activations (seq=512, batch=16): ~1.2GB
- R activations (32 slots × batch=16): ~0.1GB
- V activations: ~0.05GB
- Bridge + cross-attn: ~0.8GB
- Model weights (FP16): ~0.1GB
- Gradient checkpointing saves ~40% activation memory
- **Estimated peak per T4: ~10-12GB** — within 16GB limit with headroom

If OOM occurs: reduce batch to 8, accumulation to 16 (same effective batch).

---

## 9. Ablation Schedule

Each ablation: 200M tokens on TinyStories (~1 hour on T4×2). Total: 10 ablations = ~2 Kaggle sessions.

The ablation order is designed to validate the core components BEFORE combining them — if a component fails alone, we know not to include it.

| Day | Ablation | What we learn |
|-----|----------|---------------|
| 4 | LLaMA-style 50M baseline | BPB anchor (~1.02) |
| 4 | S only (no R, no V, typed layers) | Do typed layers alone help? |
| 5 | S + R (no adaptive K, K=1 fixed) | Does private reasoning stream help at all? |
| 5 | S + R with adaptive K | Does adaptive compute add to fixed K? |
| 6 | S + V only (no R) | Does verification alone help? |
| 6 | Full AURORA (S + R + V, adaptive K) | Core hypothesis — all streams together |
| 7 | Full AURORA vs bridge top-k=1,4,8,16 | Optimal sparsity of causal bridging |
| 8 | Full AURORA vs K_max=2 vs K_max=4 | Optimal max reasoning iterations |
| 9 | Full AURORA vs n_slots=16,32,64 | Optimal reasoning slot count |
| 10 | Full AURORA: Muon vs AdamW | Optimizer impact |

**Fallback triggers:**
- S+R (K=1) shows no improvement over baseline → reasoning stream idea fails; fall back to typed-layers-only architecture
- Adaptive K shows no improvement over fixed K=1 → drop difficulty estimator, use fixed K=2
- Full AURORA shows worse BPB than S-only → stream interactions conflict; debug Bridge Layer gating
- OOM on T4 → enable gradient checkpointing + reduce batch_size=8

---

## 10. Codebase Structure

```
nexus-lm/
├── config/
│   ├── baseline_llama.yaml
│   └── nexus_aurora_v1.yaml
├── data/
│   ├── prepare.py              # FineWeb-Edu + TinyStories download + preprocess
│   ├── tokenizer.py            # SentencePiece BPE, vocab=8192
│   └── dataloader.py           # memmap-based fast loading
├── model/
│   ├── surface.py              # SurfaceStream: Pattern + Semantic layers
│   ├── reasoning.py            # ReasoningStream: 32-slot non-causal transformer
│   ├── bridge.py               # BridgeLayer: bidirectional cross-attention
│   ├── verify.py               # VerifyLayer: surprise signal + gating
│   ├── difficulty.py           # DifficultyEstimator: K selection per position
│   ├── cope.py                 # CoPE positional encoding
│   ├── block.py                # AuroraBlock: P→Se→Difficulty→R×K→Bridge→Verify
│   ├── model.py                # NexusAurora: 6 blocks + output head
│   └── baseline.py             # LLaMA-style baseline
├── training/
│   ├── trainer.py              # Training loop, mixed precision, DDP, aux loss
│   ├── muon.py                 # Muon optimizer
│   └── scheduler.py            # WSD schedule
├── evaluation/
│   ├── perplexity.py           # BPB metric
│   ├── benchmarks.py           # lm-evaluation-harness integration
│   ├── routing_analysis.py     # K distribution, slot usage, surprise stats
│   └── ablation_runner.py      # Automated ablation orchestration
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_ablations.ipynb
│   ├── 03_full_train.ipynb
│   └── 04_evaluate.ipynb
└── results/
```

---

## 11. Evaluation Suite & Success Metrics

### 11.1 Primary Metric: BPB (Bits Per Byte)

All results reported in BPB for tokenizer-independent comparison.

### 11.2 Comparison Targets

| Model | BPB (approx) |
|-------|-------------|
| GPT-2 Small baseline (our impl, 50M) | ~1.08 |
| LLaMA-style baseline (our impl, 50M) | ~1.02 |
| Pythia-70M (published) | ~1.05 |
| **NEXUS-AURORA target** | **≤0.94** |

### 11.3 Success Criteria

**Primary win:** ≤0.94 BPB vs LLaMA baseline ~1.02 at equal training compute (same token budget).

**Secondary wins (any one sufficient):**
- Equal BPB with ≥30% fewer FLOPs (adaptive K routing working)
- Higher BPB on TinyStories ablation but with K distribution showing meaningful variation (routing is learning, not collapsed)
- Verification surprise score shows declining trend over training (V and R are converging)

**Architecture validation signals** (logged every 1000 steps):
```
k_distribution: {K=1: %, K=2: %, K=4: %}   # Is difficulty estimator active?
surprise_mean, surprise_std                  # Is V learning?
bridge_sparsity: mean active slots per token # Are causal links forming?
r_slot_entropy: entropy of slot usage        # Are reasoning slots specializing?
```

If `k_distribution` = {K=1: 100%} → difficulty estimator collapsed, not learning
If `r_slot_entropy` = maximum → slots not specializing, R is not learning distinct roles
If `surprise_mean` is flat across training → V is not learning to predict S

### 11.4 Full Evaluation Suite (Phase 1D)

| Metric | Tool |
|--------|------|
| Validation BPB | Custom, every 10K steps |
| Zero-shot: HellaSwag, ARC-Easy, PIQA, WinoGrande | lm-evaluation-harness |
| Length generalization (train 512, eval 1024/2048) | Custom |
| Inference latency & throughput | Custom |
| K distribution analysis | Custom |
| R slot specialization (cosine similarity between slots) | Custom |
| Surprise score trajectory over training | Custom |
| Peak GPU memory | torch.cuda.max_memory_allocated |

---

## 12. Breakthrough Hypotheses

### H1: Private Reasoning Improves Surface Quality
**Prediction:** S+R (K=1 fixed) achieves ≥5% BPB improvement over S-only typed architecture. The reasoning stream provides additional signal that the surface stream cannot derive alone.
**Falsification:** S+R BPB ≥ S-only BPB → reasoning stream adds no value.

### H2: Adaptive Compute Amplifies Reasoning Gains
**Prediction:** Adaptive K achieves ≥3% additional BPB improvement over fixed K=1. The K distribution shows meaningful variation (not collapsed to K=1 or K=4 for all tokens).
**Falsification:** Adaptive K = fixed K=2 in BPB → difficulty estimator learns nothing.

### H3: Verification Gate Improves Consistency
**Prediction:** Full AURORA (with V) achieves ≥2% BPB improvement over S+R without V. V's surprise score shows a declining trend over training — model becomes more self-consistent.
**Falsification:** Surprise score remains flat or random → V is not learning.

### H4: Reasoning Slots Specialize
**Prediction:** After full training, pairwise cosine similarity between the 32 R slots is lower than at initialization — slots have diverged to represent distinct concepts.
**Falsification:** Slots remain near-identical → R has collapsed to uniform representation.

### H5: Sparse Causal Bridge Creates Structured Grounding
**Prediction:** At top-k=8, Bridge Layer attention patterns show consistent structure — specific R slots reliably attend to specific token types (nouns, logical operators, numbers), more than k=32 (dense) baseline.
**Falsification:** Sparse and dense bridging achieve same BPB → sparsity provides no structural benefit.

---

## 13. What Was Removed from Original Plan (v1.0 HRD) and Why

| Removed | Reason |
|---------|--------|
| Entire HRD architecture (nGPT+DiffAttn+MoR combination) | Addresses zero fundamental limitations; combines published work without new insight |
| nGPT hypersphere normalization | Good technique, keep as optional ablation in training dynamics, not core |
| Differential Attention | Noise cancellation — useful but incremental, not a new paradigm |
| Mixture of Recursions | Parameter efficiency — good, but single-stream; doesn't enable private computation |
| CoPE as core innovation | Kept as positional encoding choice, not architectural centerpiece |
| SOAP optimizer | Muon clearly superior at small scale; redundant |
| NoPE experiment | Not relevant to AURORA's research questions |

---

## 14. Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| OOM on T4 with 3 streams | Medium | gradient_checkpointing=True + batch=8 fallback |
| R stream doesn't learn (slot collapse) | Medium | Auxiliary diversity loss: penalize high cosine similarity between slots |
| Difficulty estimator collapses to K=1 | Medium | Add entropy regularization on K distribution |
| Bridge Layer instability (cross-stream gradient interference) | Low-Medium | Gradient clipping per stream; separate optimizers for S/R/V |
| V stream produces trivial constant surprise | Medium | Initialize V weights with small positive bias; monitor during warmup |
| DDP gradient sync with variable K per position | Low | K is computed per-position within each GPU, no cross-GPU sync needed |
| Full AURORA slower than baseline (3x streams) | Certain | Expected ~2-2.5× slower training; compensate with smaller batch |
| 46M params vs 50M baseline perceived as unfair | Low | Compare on equal-FLOP basis, not parameter count |

---

## 15. Phase 2 Preview

**Trigger:** Primary success criterion met (≤0.94 BPB), H1 and H2 hypotheses validated.

**Scale-up plan:**
1. AURORA-350M: d_s=1024, d_r=512, 12 blocks, 8× A100s, ~15B tokens
   - Key question: do reasoning slots maintain specialization at scale?
   - Compare vs Pythia-410M, OPT-350M

2. AURORA-1.3B: d_s=2048, d_r=768, 16 blocks, ~50B tokens
   - Compare vs LLaMA-1.3B, TinyLlama-1.1B

3. Publish regardless — ablation results at 46M are community-valuable
4. Open-source: weights, training code, slot visualization tools, ablation CSVs

**The long-term vision:** If AURORA validates, the three-stream paradigm becomes a new architectural template — not another transformer variant, but a genuinely different structural organization of computation in language models.

---

*Living document. Update after each experimental phase.*
