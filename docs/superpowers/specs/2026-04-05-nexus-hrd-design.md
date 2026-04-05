# NEXUS-HRD: HyperRecursive Differential Transformer
## Design Specification — Project NEXUS

**Date:** 2026-04-05  
**Version:** 1.0  
**Status:** Approved for implementation planning  
**Objective:** Build a novel ~18M-parameter LLM on Kaggle T4×2 (with 32 equivalent-layer effective depth via weight sharing) that introduces a genuinely unexplored architectural combination — validating a breakthrough hypothesis before scaling to 350M–1.3B parameters. Fair comparison against 50M baselines is on equal *compute budget* (FLOPs), not parameter count.

---

## 1. Core Thesis

Three independently validated innovations from 2024–2025 — nGPT's hypersphere normalization (4–20x faster convergence), Differential Attention's noise cancellation (65% parameter efficiency), and Mixture of Recursions' adaptive recursive depth (2x inference throughput, 50% KV reduction) — have **never been combined**. Their synergies are theoretically grounded and empirically unexplored. NEXUS-HRD tests this combination as a single coherent architectural hypothesis.

**Why synergies compound rather than conflict:**
1. nGPT's unit-norm representations → smoother loss landscape → MoR routers classify on clean hypersphere surface, converging faster
2. Differential Attention cancels noise → cleaner hidden states → MoR routers make sharper easy/hard token decisions
3. Recursive weight sharing (MoR) + unit norm (nGPT) → shared weights see geometrically consistent inputs at every recursion depth, eliminating scale drift
4. All three operate on normalized representations → consistent gradient flow throughout

---

## 2. Architecture: HyperRecursive Differential Transformer (HRD)

### 2.1 Overview

NEXUS-HRD is a **weight-shared recursive transformer**. A single stack of `L=8` layers is executed up to `R=4` times (recursion steps). At each recursion step, a lightweight per-token router decides which tokens continue to the next step and which exit early. Easy tokens exit after 1–2 recursions; hard tokens use all 4. Within each layer, attention uses the differential mechanism. All hidden states and weight matrices are constrained to the unit hypersphere.

```
Effective depth = 32 equivalent layers (L=8 × R=4)
Actual parameter count ≈ 50M (weight sharing prevents parameter explosion)
Active parameters per forward pass: varies by token routing (avg ~60–70% of full depth)
```

### 2.2 Innovation 1 — nGPT Hypersphere Normalization

Every vector in the model lives on the unit hypersphere (L2 norm = 1): token embeddings, weight matrix rows, hidden states after each sublayer, Q/K/V projections.

Residual update replaces both LayerNorm and standard residual connection:

```
h_new = normalize(h + α · (sublayer_output - h))
```

Where `α` is a learnable scalar per sublayer, initialized to 0.1. Matrix-vector multiplications become cosine similarities bounded in [-1, 1], eliminating the need for weight decay and providing intrinsic training stability.

**Why this helps MoR:** When a token re-enters the same shared layer at recursion step 2, its hidden state is on the same hypersphere as step 1. The shared weights see geometrically consistent inputs regardless of recursion depth — no scale drift across recursion steps.

**Implementation note:** `normalize()` runs in FP32, then casts back to FP16. This is the only FP32 operation beyond loss computation.

### 2.3 Innovation 2 — Differential Attention

Each attention head computes two softmax maps over the same sequence, then subtracts them with a learned scalar `λ`:

```
DiffAttn(Q, K, V) = (softmax(Q₁Kᵀ/√d) - λ · softmax(Q₂Kᵀ/√d)) · V
```

`Q` is split into `Q₁, Q₂` (half dimensions each). `λ` is initialized to 0.8 and learned per head. The subtraction cancels uniform/noisy attention weights, amplifying only strongly-attended tokens.

**Configuration:** `n_heads=8` in standard terms → 4 effective differential heads (each uses both Q halves). Head dimension `d_head = d_model / n_heads = 64`. Since Q is split into Q₁ and Q₂, each half has dimension 32. The K and V projections remain full 64-dim per head. This is consistent with the original DiffAttn paper's parameterization.

**Why this helps MoR:** Cleaner attention maps produce more informative hidden states. The token router receives cleaner signals about which tokens have fully integrated their context (easy → exit) vs. which still have unresolved dependencies (hard → continue recursing).

**QK-Norm:** Not needed. nGPT already normalizes Q and K on the hypersphere — this replaces QK-Norm entirely.

### 2.4 Innovation 3 — Mixture of Recursions (MoR) Token Routing

After each recursion step, a router MLP scores each active token:

```
score = sigmoid(router_r(h_token))  ∈ [0, 1]
continue = score > threshold_r       (threshold is a learned parameter per step r)
```

Tokens below threshold exit; their last hidden state is used for prediction. Active tokens pass into the next recursion step with the same shared `HRDBlock` weights.

**Router architecture:** 2-layer MLP, `D → D/4 → 1`, with SiLU activation. One router instance per recursion step (routers are NOT weight-shared — each step needs its own routing logic). ~0.5M total router parameters.

**KV cache behavior:** Only active tokens are included in KV cache at each step. Memory shrinks as tokens exit. At R=4 with healthy routing (~30% active at final step), effective KV cache is ~60% smaller than a standard 32-layer model.

**Training:** Straight-through estimator for gradient flow through discrete routing decisions. Auxiliary load-balancing loss prevents routing collapse.

**Routing health targets:**
```
active_ratio_r1 ≈ 0.80  (20% easy tokens exit after step 1)
active_ratio_r2 ≈ 0.50  (another 30% exit)
active_ratio_r3 ≈ 0.30  (another 20% exit)
active_ratio_r4 ≈ 0.30  (hardest 30% use all 4 steps)
```

### 2.5 Positional Encoding — CoPE

Contextual Position Encoding (CoPE) replaces RoPE. Instead of fixed per-position biases, CoPE computes positions as weighted combinations of context vectors, conditioned on the query:

```
gate_i = softmax(q · e_i)           # attention over 16 context position vectors
pos = Σ gate_i · position_embed_i   # contextual position
```

This gives each token a position that depends on what it's attending to, not just its index. Proven 5–17% perplexity improvement over RoPE, and naturally generalizes to sequences longer than training length (tested: train 512, eval 2048 with <10% degradation).

**Configuration:** 16 context position vectors (`cope_positions=16`).

**nGPT compatibility:** CoPE position vectors must also live on the unit hypersphere (L2-normalized). The positional signal is added to Q before the nGPT normalization step in DiffAttn, so position information is incorporated into the normalized query without breaking the hypersphere constraint.

### 2.6 FFN — SwiGLU

Standard SwiGLU FFN with 2x width ratio (narrower than the typical 4x, compensated by recursive depth):

```
FFN(x) = (W₂ · (SiLU(W₁x) ⊙ W₃x))
d_ffn = 2 × d_model = 1024
```

No MoE, no sparse activation at Phase 1 scale. These are Phase 2 explorations if Phase 1 validates the core HRD hypothesis.

### 2.7 Full Layer (HRDBlock) — Execution Order

```
Input: x ∈ ℝ^(B × T_active × D), on unit hypersphere

1. attn_out = DifferentialAttention(x, cope_pos)
2. x = normalize(x + α_attn · (attn_out - x))     # nGPT update
3. ffn_out = SwiGLU(x)
4. x = normalize(x + α_ffn · (ffn_out - x))        # nGPT update

Output: x ∈ ℝ^(B × T_active × D), on unit hypersphere
```

Parallel Attention+FFN (PaLM-style) is NOT used. Sequential order is maintained for compatibility with nGPT's interpolation geometry.

---

## 3. Model Dimensions & Parameter Budget

| Component | Parameters | Notes |
|-----------|------------|-------|
| Token embedding (8192 × 512) | 4.2M | Tied with output projection |
| Output projection (512 × 8192) | 0M | Tied with embedding |
| HRDBlock × 1 shared | ~13M | Attn + FFN, used 4× via recursion |
| Routers × 4 (one per recursion step) | ~0.5M | Not shared |
| CoPE position vectors | ~0.008M | Negligible |
| **Total** | **~18M** | |

**Note:** At 18M parameters, NEXUS-HRD is dramatically smaller than a 50M baseline — yet through 4 recursion steps, it performs the equivalent computation of a 32-layer model. This is the central efficiency claim: fewer parameters, deeper effective computation, adaptive cost per token.

If 18M feels too small for a fair comparison, the model can be scaled up to `d_model=768` for ~40M total, keeping the same architectural structure. This is a hyperparameter decision for Day 4 after seeing baseline performance.

---

## 4. Training Configuration

### 4.1 Dataset

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| FineWeb-Edu (subset) | ~1.5B | Main training (high quality, curated) |
| TinyStories | ~500M | Ablation sanity checks (fast, clean signal) |
| Validation | 1% of FineWeb-Edu | Perplexity tracking |

Pre-tokenized to `train.bin`, `val.bin` (np.memmap, uint16, BPE vocab=8192).

### 4.2 Hyperparameters

```yaml
model:
  d_model: 512
  n_heads: 8              # 4 effective differential heads
  d_ffn: 1024             # 2× ratio
  n_layers: 8             # shared across R steps
  max_recursion: 4        # R=4 maximum steps
  vocab_size: 8192
  max_seq_len: 512
  cope_positions: 16

training:
  batch_size: 32                    # per GPU
  gradient_accumulation: 4          # effective batch = 256
  max_lr: 3e-3                      # Muon optimal
  min_lr: 3e-4                      # WSD final LR
  warmup_tokens: 100_000_000        # 100M tokens
  stable_tokens: 1_400_000_000      # 1.4B tokens
  decay_tokens: 100_000_000         # 100M tokens sharp decay
  grad_clip: 1.0
  weight_decay: 0.0                 # nGPT unit-norm replaces weight decay entirely
  dtype: float16                    # T4 requirement

optimizer:
  name: muon
  momentum: 0.95
  nesterov: true
  embed_lr: 3e-4           # embeddings + routers use AdamW
  router_lr: 1e-3          # higher LR for discrete decision makers

router:
  init_threshold: 0.5
  load_balance_coeff: 0.01
  min_active_ratio: 0.2    # force ≥20% tokens to always recurse

init:
  scheme: u-muP
  alpha_init: 0.1          # nGPT interpolation scalars
  lambda_init: 0.8         # DiffAttn noise cancellation scalars
  lambda_min: 0.0

hardware:
  gpus: 2                  # Kaggle T4 × 2
  strategy: ddp
  gradient_checkpointing: false
```

### 4.3 Optimizer — Muon

Muon applies Newton-Schulz orthogonalization to gradient matrices. It naturally keeps weight matrices close to orthogonal — complementing nGPT's unit-norm constraint. This is an additional synergy beyond the primary three. Muon achieves 2× compute efficiency vs AdamW at small scale (confirmed by Moonlight 3B, 2025). Embedding and router parameters use AdamW (Muon only applies to matrix parameters).

### 4.4 Schedule — WSD (Warmup-Stable-Decay)

Three phases:
- **Warmup (100M tokens):** Linear LR increase to max_lr. Routers stabilize, nGPT α scalars settle.
- **Stable (1.4B tokens):** Constant max_lr. Main learning phase. Muon benefits most here.
- **Decay (100M tokens):** Sharp cosine decay to min_lr. Final loss compression.

WSD is mathematically proven optimal for LLM pretraining (ICML 2025) and is now the de facto standard over cosine decay.

### 4.5 Initialization — u-μP

Unit-Scaled Maximal Update Parametrization (u-μP, ICLR 2025). Provides:
- Better default hyperparameters than standard μP
- Clean FP16 training (unit-scaled activations)
- Hyperparameter transfer: optimal LR found at 50M transfers to 500M+ without retuning (Phase 2 prerequisite)

---

## 5. Ablation Schedule

Each ablation trains for 200M tokens (~1 hour on T4×2). 10 runs total = ~2 Kaggle sessions.

| Day | Ablation | Hypothesis Tested |
|-----|----------|------------------|
| 4 | LLaMA-style baseline | Anchor BPB |
| 4 | + DiffAttn only | Noise cancellation at 50M scale |
| 5 | + nGPT only | Convergence speedup |
| 5 | + MoR only (standard attn) | Recursive routing benefit |
| 6 | DiffAttn + nGPT | Primary synergy: clean geometry + clean attention |
| 6 | nGPT + MoR | Secondary synergy: sphere geometry helps routing |
| 7 | Full HRD (all three) | Triple synergy — core hypothesis |
| 8 | CoPE vs RoPE in full HRD | Best positional encoding for this arch |
| 9 | Muon vs AdamW in full HRD | Optimizer synergy with nGPT geometry |
| 10 | R=2, R=3, R=4, R=6 | Optimal recursion depth |

**Fallback triggers:**
- nGPT alone shows <2x convergence speedup at 100M tokens → drop nGPT, fall back to DiffAttn+MoR (Option B)
- DiffAttn+nGPT worse than either alone → they conflict; drop DiffAttn, use standard attention with nGPT+MoR
- MoR routing collapses (all active_ratio = 1.0) → switch to Mixture of Depths (layer-level routing)

---

## 6. Codebase Structure

```
nexus-lm/
├── config/
│   ├── baseline_llama.yaml
│   └── nexus_hrd_v1.yaml
├── data/
│   ├── prepare.py
│   ├── tokenizer.py
│   └── dataloader.py
├── model/
│   ├── hypersphere.py      # normalize(), slerp_update(), nGPT ops
│   ├── attention.py        # DifferentialAttention
│   ├── cope.py             # CoPE positional encoding
│   ├── ffn.py              # SwiGLU FFN
│   ├── router.py           # MoRRouter + routing logic + aux loss
│   ├── block.py            # HRDBlock (one layer)
│   ├── model.py            # NexusHRD recursive execution engine
│   └── baseline.py         # LLaMA-style baseline
├── training/
│   ├── trainer.py          # Training loop, mixed precision, DDP
│   ├── muon.py             # Muon optimizer
│   └── scheduler.py        # WSD schedule
├── evaluation/
│   ├── perplexity.py       # BPB metric
│   ├── benchmarks.py       # lm-evaluation-harness integration
│   └── ablation_runner.py  # Automated ablation orchestration
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_ablations.ipynb
│   ├── 03_full_train.ipynb
│   └── 04_evaluate.ipynb
└── results/
```

---

## 7. Evaluation Suite & Success Metrics

### 7.1 Primary Metric: BPB (Bits Per Byte)

All perplexity values converted to BPB for tokenizer-independent comparison across models with different vocabulary sizes.

### 7.2 Comparison Targets

| Model | BPB (approx) |
|-------|-------------|
| GPT-2 Small baseline (50M, our impl) | ~1.08 |
| LLaMA-style baseline (50M, our impl) | ~1.02 |
| Pythia-70M (published) | ~1.05 |
| **NEXUS-HRD target** | **≤0.94** |

### 7.3 Success Criteria

**Primary win:** NEXUS-HRD achieves ≤0.94 BPB vs LLaMA baseline ~1.02 at equal training compute.

**Secondary win (any one sufficient):**
- Equal BPB with ≥40% fewer active parameters per forward pass (MoR routing confirmed)
- Equal BPB in ≤60% of training tokens (nGPT convergence speedup confirmed)
- ≥1.5× inference throughput at equal quality (MoR exit routing confirmed)

**Phase 2 trigger:** Primary win + healthy routing + u-μP transfer confirmed at proxy model.

### 7.4 Metrics Logged Per Run

```
step, train_loss, val_loss, bpb, lr, tokens_per_sec, gpu_mem_mb,
wall_time, grad_norm, active_tokens_r1, active_tokens_r2,
active_tokens_r3, active_tokens_r4, lambda_mean, alpha_mean
```

`active_tokens_r*`: routing health monitoring  
`lambda_mean`: DiffAttn noise cancellation scalar — early instability warning  
`alpha_mean`: nGPT interpolation scalar — convergence speed indicator

### 7.5 Evaluation Suite (Phase 1D)

| Metric | Tool |
|--------|------|
| Validation BPB | Custom, every 10K steps |
| Zero-shot: HellaSwag, ARC-Easy, PIQA, WinoGrande | lm-evaluation-harness |
| Length generalization (train 512, eval 1024/2048) | Custom |
| Inference latency & throughput | Custom |
| Routing entropy per step | Custom |
| Peak GPU memory | torch.cuda.max_memory_allocated |

---

## 8. What Was Removed From Original Plan (and Why)

| Removed | Reason |
|---------|--------|
| LayerNorm, RMSNorm, Sandwich-LN, DeepNorm ablations | Replaced entirely by nGPT interpolation — different paradigm |
| QK-Norm | nGPT already normalizes Q and K on hypersphere |
| Post-LN and Pre-LN comparison | Irrelevant to nGPT architecture |
| Standard MHA and GQA | Replaced by Differential Attention |
| NoPE experiment | Replaced by CoPE (proven superior, more interesting) |
| SOAP optimizer ablation | Muon clearly superior at small scale per 2025 results |
| Byte-level tokenization experiment | Sequence length explosion on T4s (4× longer sequences) |
| Dynamic vocabulary | Too experimental, high implementation risk, low payoff |
| DataParallel option | DDP only — better gradient sync efficiency |
| Parallel Attention+FFN (PaLM-style) | Incompatible with nGPT's interpolation geometry |
| Hierarchical tokenization | Too complex for Phase 1, separate research track |

---

## 9. Breakthrough Hypotheses (Revised)

### H1: Triple Synergy — Core Hypothesis
**Prediction:** nGPT + DiffAttn + MoR together achieves ≥12% BPB reduction over LLaMA baseline at equal compute. Each pairwise combination achieves ≥7%.  
**Falsification:** If full HRD underperforms the best pairwise combination, interactions conflict rather than synergize.

### H2: Hypersphere Geometry Improves Routing
**Prediction:** nGPT + MoR achieves better routing (lower routing entropy, more selective active_ratio) than MoR with standard normalization. Clean geometry → sharper routing decisions.  
**Falsification:** If active_token_ratios are identical between nGPT+MoR and standard+MoR.

### H3: Muon-nGPT Geometric Synergy
**Prediction:** Muon's orthogonalization + nGPT's unit-norm gives a 10–15% additional convergence speedup vs AdamW+nGPT. Both push weights toward orthogonal/unit structure.  
**Falsification:** If Muon and AdamW show equal convergence speed in the full HRD model.

### H4: u-μP Hyperparameter Transfer
**Prediction:** Optimal LR found at 50M transfers to 500M without retuning. Proxy model (13M) achieves same optimal LR as 50M model.  
**Falsification:** If proxy model optimal LR diverges from 50M optimal LR by >2×.

---

## 10. Phase 2 Preview

**Trigger:** Phase 1 primary success criteria met.

1. Scale to 350M parameters on 4–8× A100/H100
   - d_model=1024, n_layers=16, R=4, full FineWeb-Edu (~15B tokens)
   - Compare: Pythia-410M, OPT-350M, GPT-2 Medium
   - Key test: do HRD innovations hold at 7× scale?

2. Scale to 1.3B if 350M results hold
   - Train on 50–100B tokens (Chinchilla-optimal)
   - Compare: LLaMA-1.3B, TinyLlama-1.1B, Pythia-1.4B

3. Publish regardless of outcome — ablation results at small scale are community-valuable

4. Open-source: model weights, training code, all ablation CSVs

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| nGPT and DiffAttn conflict (both normalize differently) | Day 6 pairwise ablation detects this before full model commit |
| MoR routing collapse on T4 with variable-length tensors | Pad to max active length within GPU, masked attention |
| nGPT FP16 instability (normalization in FP16) | normalize() always runs in FP32, cast back |
| DDP + variable active tokens mismatch across GPUs | No cross-GPU token count sync during forward pass |
| u-μP proxy test fails (no transfer) | Fall back to standard hyperparameter search; still valid Phase 1 |
| Kaggle session timeout mid-run | Checkpoint every 50M tokens; training is resumable |
| 18M parameter count too small for fair comparison | Scale to d_model=768 (~40M) if baseline gap is too large |

---

*Living document. Update after each experimental phase with findings, revised hypotheses, and adjusted plans.*
