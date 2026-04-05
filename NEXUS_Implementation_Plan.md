# Project NEXUS: Novel EXperimental Unified System
## Implementation Plan — Building a Next-Generation LLM from Scratch

**Version:** 1.0  
**Date:** April 2026  
**Objective:** Guide an advanced AI (Claude Opus 4.6) through a systematic research-driven process to architect, build, train, and validate a novel small-scale language model that introduces architectural breakthroughs — then scale up after validation.

---

## 1. Executive Summary

This plan outlines a phased approach to building a language model from scratch that challenges existing architectural assumptions. Rather than tweaking hyperparameters on a standard Transformer, we systematically explore fundamental design changes at the architecture level — attention mechanisms, positional encoding, normalization strategies, activation functions, tokenization, and training dynamics — to discover configurations that yield better performance per FLOP than current state-of-the-art models.

**Phase 1 (this document's focus):** Build a ~25M–60M parameter model trainable within Kaggle's free tier (2× T4 GPUs, 16GB VRAM each, ~30GB system RAM, 11-hour session limit). Compare against identically-sized baselines (GPT-2 Small architecture, LLaMA-style architecture) on standardized benchmarks.

**Phase 2 (contingent on Phase 1 success):** Scale to 350M–1.3B parameters on dedicated GPU clusters and validate that improvements hold at scale.

---

## 2. Kaggle Free Tier — Hard Constraints

Every design decision in Phase 1 must respect these constraints:

| Resource | Limit |
|----------|-------|
| GPU | 2× NVIDIA T4 (16GB VRAM each) |
| System RAM | ~30GB |
| Disk | ~70GB (including OS) |
| Session time | ~11 hours continuous |
| Internet | Available during setup, can be turned off for training |
| Persistent storage | Kaggle Datasets (up to 100GB upload) |

**Derived training budget:**
- At mixed precision (FP16) on T4, expect ~20–40 TFLOPS effective.
- For a 50M parameter model with sequence length 512 and batch size 32, one forward+backward pass ≈ 0.3 seconds.
- In 10 hours of pure training: ~120,000 steps → ~3.8M sequences → ~2B tokens (single epoch over a curated subset).
- Target: Train on 1–2B tokens from a curated, high-quality dataset (e.g., a cleaned subset of SlimPajama, TinyStories, or OpenWebText).

---

## 3. Research Directives for the AI Architect

The guiding AI (Claude Opus 4.6 or equivalent) should approach this as a research scientist. Below are the specific investigation areas, ordered by expected impact.

### 3.1 Attention Mechanism Innovations

**Current limitation:** Standard multi-head attention (MHA) scales O(n²) in sequence length and treats all heads uniformly.

**Directives for the AI:**

1. **Investigate Differential Attention** — Recent work (Microsoft's Diff Transformer, late 2024) subtracts two softmax attention maps to cancel noise and amplify signal. The AI should study this mechanism, understand its mathematical formulation, and implement it. Key question: Does differential attention yield better perplexity at the 50M scale compared to standard MHA?

2. **Explore Multi-Scale Attention Heads** — Instead of uniform head dimensions, design heads that operate at different granularities. Some heads attend to local context (window=64), some to medium range (window=256), and some to full sequence. This is a hybrid of local attention and global attention but within a single layer.

3. **Investigate Linear Attention Variants** — Study kernelized attention (Performers, Random Feature Attention) and recent linear recurrence models (RetNet, Mamba-style selective state spaces). Can we hybridize: use linear attention for most layers and full quadratic attention only in strategic layers (e.g., every 4th layer)?

4. **Attention Head Pruning During Training** — Implement a learnable gate per head that can drive heads to zero during training. This automatically discovers how many heads each layer actually needs. Compare final model quality vs. fixed-head configurations.

**Deliverable:** A custom attention module (`NexusAttention`) that implements the best-performing variant from the above explorations. Must include ablation results.

### 3.2 Positional Encoding Breakthroughs

**Current limitation:** RoPE (Rotary Position Embeddings) is dominant but has known issues with length generalization. ALiBi is simpler but less expressive.

**Directives for the AI:**

1. **Study Learnable Frequency RoPE** — Instead of fixed frequency bands in RoPE (θ_i = 10000^{-2i/d}), make the base frequencies learnable parameters. Initialize them at standard RoPE values but allow gradient updates. Does the model learn better frequency patterns than the hand-designed ones?

2. **Investigate Position-Aware Gating** — After computing attention with positional encoding, apply a learned gate that modulates how much positional information flows through. Hypothesis: Some layers need strong positional signal (syntax), others need more semantic (meaning-focused) attention.

3. **Explore Hybrid Positional Schemes** — Use ALiBi-style linear bias in early layers (where local/syntactic patterns dominate) and RoPE in later layers (where long-range semantic relationships matter).

4. **Test NoPE (No Positional Encoding)** — Some recent work suggests that with causal masking alone, models can implicitly learn positional information. Test this at small scale.

**Deliverable:** A `NexusPosition` module with the best-performing scheme. Must include length generalization tests (train on 512, evaluate on 1024 and 2048).

### 3.3 Normalization and Residual Connections

**Current limitation:** Pre-LayerNorm (Pre-LN) is standard. RMSNorm is common for efficiency. But normalization placement and type significantly affect training dynamics.

**Directives for the AI:**

1. **Implement and Compare:**
   - Pre-LN (standard)
   - Post-LN (original Transformer)
   - Sandwich-LN (norm before AND after sublayer)
   - DeepNorm (Microsoft, scaled residuals for very deep models)
   - QK-Norm (normalize queries and keys before attention)
   - RMSNorm variants

2. **Explore Adaptive Normalization** — A learned interpolation between LayerNorm and RMSNorm per layer. Let the model discover which layers benefit from the mean-centering of LayerNorm vs. the simplicity of RMSNorm.

3. **Residual Scaling** — Instead of uniform residual connections (x + sublayer(x)), use learned per-layer scaling: x + α_l · sublayer(x), where α_l is a learned scalar initialized to 1.0. This gives the model control over how much each layer contributes.

4. **Sub-Layer Reordering** — The standard order is: Norm → Attention → Residual → Norm → FFN → Residual. Explore parallel attention+FFN (as in PaLM) where attention and FFN run in parallel: x + Attention(Norm(x)) + FFN(Norm(x)). This can be ~15% faster.

**Deliverable:** A `NexusBlock` (single transformer layer) implementing the optimal combination.

### 3.4 Feed-Forward Network (FFN) Redesign

**Current limitation:** Standard FFN uses two linear projections with a nonlinearity. SwiGLU (LLaMA-style) is current best practice but may not be optimal.

**Directives for the AI:**

1. **Benchmark Activation Functions at Small Scale:**
   - ReLU, GELU, SiLU/Swish
   - SwiGLU, GeGLU, ReGLU
   - Squared ReLU (shown to work well in some settings)
   - Softmax-based activations (less explored)
   - Novel: Learnable activation functions (parameterized as small neural networks)

2. **Explore Mixture-of-Experts (MoE) at Small Scale** — Even at 50M total parameters, can a simple top-2-of-8 MoE FFN outperform a dense FFN? The trick is: each expert is small (8 experts × fraction of FFN size), so total params might be similar but active params per token are lower. This tests whether MoE benefits exist at very small scale.

3. **Investigate FFN Width vs. Depth Tradeoff** — Standard ratio is FFN hidden dim = 4× model dim. Test: What if we use 2× width but add a second FFN sublayer per block? More depth, less width per layer.

4. **Sparse Activation in FFN** — Implement top-k activation after the first linear layer (only keep the k largest activations, zero the rest). This creates implicit sparsity without a routing mechanism. Test k = 50%, 25%, 10% of hidden dim.

**Deliverable:** Optimal FFN configuration with benchmarks.

### 3.5 Tokenization Innovation

**Current limitation:** BPE (Byte-Pair Encoding) is standard. SentencePiece with BPE or Unigram models dominate. But tokenization fundamentally limits what the model can learn.

**Directives for the AI:**

1. **Byte-Level Processing** — Instead of subword tokenization, operate directly on UTF-8 bytes (vocabulary size = 259: 256 bytes + special tokens). This eliminates tokenization artifacts and makes the model truly language-agnostic. Challenge: Sequences become ~4× longer. Mitigation: Use a byte-level encoder (small CNN or 1D convolution) that compresses byte sequences into a shorter sequence of embeddings before feeding into the Transformer.

2. **Hierarchical Tokenization** — Two-stage embedding: first embed individual characters/bytes, then use a small local Transformer or CNN to create "word-level" embeddings from character sequences. The main model then operates on these compressed representations.

3. **Dynamic Vocabulary** — Train a standard BPE tokenizer but allow the model to merge or split tokens during training based on how useful they are (measured by gradient signal to their embeddings).

4. **Practical Recommendation** — For Kaggle-scale, start with a standard BPE tokenizer (vocab size 8192–16384, much smaller than typical 32K–128K) trained on the training corpus. Smaller vocab = smaller embedding tables = more parameter budget for the core model. Test whether this hurts or helps at the 50M scale.

**Deliverable:** Tokenizer configuration and analysis of vocab size impact on model quality.

### 3.6 Training Dynamics and Optimization

**Directives for the AI:**

1. **Learning Rate Schedule Innovation:**
   - Standard: Linear warmup + cosine decay
   - Test: WSD (Warmup-Stable-Decay), which holds LR constant for most of training then decays sharply. Recent evidence suggests this can match or beat cosine in less wall-clock time.
   - Test: Cyclical learning rates (but constrained to not require extra epochs)

2. **Optimizer Selection:**
   - AdamW (baseline)
   - SOAP (recent optimizer claimed to outperform Adam on LLM training)
   - Muon/Shampoo-style second-order optimizers (may converge faster per step)
   - Implement and compare at small scale — optimizer choice can change optimal hyperparameters

3. **Batch Size Scheduling** — Start with small batch size (high LR-to-batch ratio for fast initial learning), gradually increase batch size during training (equivalent to LR decay but maintains gradient noise benefits longer).

4. **Gradient Clipping Strategy** — Standard global norm clipping at 1.0. Test per-layer adaptive clipping where the clip threshold is set per-layer based on running statistics.

5. **Weight Initialization** — Beyond standard Xavier/He init, explore μP (Maximal Update Parametrization) which provides hyperparameter transfer across model scales. If we find good hyperparameters at 50M, μP predicts they'll work at 500M+.

**Deliverable:** Full training configuration with ablation results.

### 3.7 Architecture-Level Structural Innovations

**Directives for the AI:**

1. **Layer Depth vs. Width Analysis** — For a fixed 50M parameter budget, is it better to have:
   - 12 layers × 512 dim (standard)
   - 24 layers × 384 dim (deeper, narrower)
   - 6 layers × 768 dim (shallower, wider)
   - Compare perplexity AND inference speed

2. **Parameter Sharing** — Test cross-layer parameter sharing strategies:
   - Full sharing (ALBERT-style): All layers share the same weights
   - Cycle sharing: Layer weights repeat in a cycle (layers 1-4 repeated 3×)
   - Partial sharing: Attention weights shared, FFN weights unique (or vice versa)
   - This can dramatically reduce actual parameter count while maintaining effective depth

3. **Skip Connections Across Layers** — Instead of only connecting layer L to layer L+1, add DenseNet-style connections where each layer receives input from ALL previous layers (with learned gating to prevent information overload).

4. **Mixture of Depths** — Recent work on early exit and adaptive computation. Implement token-level routing where "easy" tokens skip some layers. Even at small scale, this tests whether the mechanism works.

5. **Embedding Tying** — Always tie input and output embeddings (standard practice, saves parameters). But also test: using a smaller embedding dimension and projecting up to model dimension (factorized embedding as in ALBERT).

**Deliverable:** Optimal architecture specification document.

---

## 4. Implementation Roadmap

### Phase 1A: Foundation (Days 1–3)

**Objective:** Set up the codebase, implement baselines, prepare data.

**Tasks:**

1. **Project Structure:**
```
nexus-lm/
├── config/
│   ├── baseline_gpt2.yaml      # GPT-2 Small equivalent config
│   ├── baseline_llama.yaml     # LLaMA-style baseline config
│   └── nexus_v1.yaml           # Our experimental config
├── data/
│   ├── prepare.py              # Data download + preprocessing
│   ├── tokenizer.py            # Custom tokenizer training
│   └── dataloader.py           # Efficient data loading
├── model/
│   ├── attention.py            # NexusAttention implementations
│   ├── position.py             # Positional encoding variants
│   ├── ffn.py                  # FFN variants (SwiGLU, MoE, sparse)
│   ├── block.py                # NexusBlock (single transformer layer)
│   ├── model.py                # Full NexusLM model
│   └── baselines.py            # GPT-2 and LLaMA baselines for comparison
├── training/
│   ├── trainer.py              # Training loop with mixed precision
│   ├── optimizer.py            # Custom optimizers (SOAP, etc.)
│   ├── scheduler.py            # LR schedulers
│   └── distributed.py         # Multi-GPU setup for Kaggle T4×2
├── evaluation/
│   ├── perplexity.py           # Perplexity on held-out data
│   ├── benchmarks.py           # Zero-shot benchmark evaluation
│   ├── efficiency.py           # FLOPS counting, throughput measurement
│   ├── length_generalization.py # Test on longer sequences than training
│   └── ablation.py             # Automated ablation study runner
├── notebooks/
│   ├── kaggle_train_baseline.ipynb
│   ├── kaggle_train_nexus.ipynb
│   └── kaggle_evaluate.ipynb
├── results/
│   └── (auto-generated comparison tables and plots)
└── README.md
```

2. **Data Preparation:**
   - Download a high-quality subset: TinyStories (for sanity checks, ~500M tokens) + OpenWebText subset or SlimPajama subset (~1–2B tokens for main training)
   - Train BPE tokenizer (vocab size 8192) on training data
   - Pre-tokenize and shard into binary format (np.memmap or similar) for fast loading
   - Create validation and test splits (1% each)

3. **Implement Baselines:**
   - GPT-2 Small equivalent: 12 layers, 768 dim, 12 heads, ~124M params → scale down to ~50M by reducing dim to 512 and layers to 8
   - LLaMA-style equivalent: Same param budget but with RoPE, SwiGLU, RMSNorm, GQA
   - Both must produce: (a) final validation perplexity, (b) tokens/second throughput, (c) total FLOPS consumed

### Phase 1B: Systematic Ablation Studies (Days 4–10)

**Objective:** Run controlled experiments to find the best component for each subsystem.

**Methodology:** Change ONE component at a time from the LLaMA-style baseline. Each ablation trains for the same number of tokens (200M tokens, ~1 hour on Kaggle). Measure validation perplexity.

**Ablation Schedule:**

| Day | Component | Variants to Test | Hours Needed |
|-----|-----------|-------------------|--------------|
| 4 | Attention | Standard MHA, GQA, Differential Attention, Multi-scale heads | 4h (4 runs × 1h) |
| 5 | Positional Encoding | RoPE, Learnable-freq RoPE, ALiBi, Hybrid ALiBi+RoPE, NoPE | 5h (5 runs × 1h) |
| 6 | Normalization + Residual | Pre-LN+RMSNorm, Sandwich-LN, DeepNorm, QK-Norm, Parallel Attn+FFN | 5h (5 runs × 1h) |
| 7 | FFN Design | SwiGLU, GeGLU, Squared ReLU, Sparse top-k FFN, Mini-MoE | 5h (5 runs × 1h) |
| 8 | Architecture Shape | Deep-narrow, Wide-shallow, Standard, Param-sharing cycle | 4h (4 runs × 1h) |
| 9 | Training Dynamics | AdamW+Cosine, AdamW+WSD, SOAP+Cosine, Batch size schedule | 4h (4 runs × 1h) |
| 10 | Tokenizer | Vocab 4K, 8K, 16K, 32K | 4h (4 runs × 1h) |

**Total Kaggle sessions needed:** ~7 sessions of ≤5 hours each (well within weekly limits if sessions are managed carefully).

**Important:** Log ALL metrics to CSV/JSON per run: step, train loss, val loss, learning rate, throughput (tokens/sec), GPU memory usage, wall-clock time.

### Phase 1C: Combine Best Components — Nexus v1 (Days 11–14)

**Objective:** Assemble the winning configuration and train the full model.

**Tasks:**

1. **Architecture Assembly** — Combine the best-performing variant from each ablation into the NexusLM model. Example (hypothetical best combo):
   - Differential Attention with multi-scale heads
   - Learnable-frequency RoPE
   - Sandwich-LN with QK-Norm
   - Parallel Attention+FFN layout
   - SwiGLU with sparse top-k activation
   - Deep-narrow shape (more layers, smaller dim)
   - WSD learning rate schedule with SOAP optimizer

2. **Full Training Run** — Train on the full 1–2B token budget (~8–10 hours on Kaggle). Save checkpoints every 100M tokens.

3. **Second Iteration** — If the combined model underperforms expectations (some combinations may conflict), run a quick grid search on the top-2 options for each conflicting component (Day 13–14).

### Phase 1D: Comprehensive Evaluation (Days 15–18)

**Objective:** Rigorously compare Nexus v1 against baselines.

**Evaluation Suite:**

| Metric | What It Measures | How |
|--------|-----------------|-----|
| Validation Perplexity | Raw language modeling quality | Held-out split of training distribution |
| BPB (Bits Per Byte) | Tokenizer-independent quality metric | Convert perplexity to bits per byte for fair comparison across different vocab sizes |
| Throughput (tokens/sec) | Training efficiency | Measure on identical hardware |
| FLOPS-to-quality ratio | Compute efficiency | Total training FLOPS vs. final perplexity |
| Zero-shot accuracy | Downstream task quality | HellaSwag, ARC-Easy, PIQA, WinoGrande (use lm-evaluation-harness) |
| Length generalization | Robustness | Evaluate on 2× and 4× training sequence length |
| Memory footprint | Deployment efficiency | Peak GPU memory during inference |
| Inference latency | Real-world speed | Time-to-first-token and tokens/sec generation |
| Parameter efficiency | Quality per parameter | Perplexity normalized by parameter count |
| Training stability | Reliability | Loss curve smoothness, gradient norm statistics |

**Comparison Targets:**
- Our GPT-2-style 50M baseline
- Our LLaMA-style 50M baseline
- Published results for models of similar size (GPT-2 Small numbers from the original paper, Pythia 70M, TinyLlama extrapolations)

**Success Criteria for Phase 1:**
- Nexus v1 achieves ≥5% lower perplexity than the best baseline at equal compute
- OR achieves equal perplexity with ≥20% fewer FLOPS
- OR achieves equal perplexity with ≥30% fewer parameters
- AND demonstrates stable training (no loss spikes, monotonic improvement)
- AND maintains or improves inference throughput

---

## 5. Breakthrough Hypotheses — Ranked by Expected Impact

The AI architect should pursue these as specific hypotheses to test. Each has a clear prediction and falsification criterion.

### Hypothesis 1: Differential Attention + Sparse FFN Synergy
**Prediction:** Differential attention cancels noise in attention maps, meaning the FFN receives cleaner signals. Combined with sparse top-k FFN activation, this creates a model that is both more selective in what it attends to AND more selective in how it processes attended information. Expected: ≥8% perplexity reduction at equal FLOPS.  
**Falsification:** If the combination performs worse than either component alone, the hypothesis is wrong — noise cancellation and sparsity may conflict.

### Hypothesis 2: Learnable RoPE Frequencies Improve Length Generalization
**Prediction:** Fixed RoPE frequencies are suboptimal because different heads need different frequency patterns. Learnable frequencies will reduce perplexity by ≥3% on in-distribution AND show <10% perplexity degradation at 2× sequence length (vs. >30% for fixed RoPE).  
**Falsification:** If learnable frequencies converge to values near the fixed defaults, there's no benefit.

### Hypothesis 3: Parallel Attention+FFN with Deeper Networks
**Prediction:** Running attention and FFN in parallel (PaLM-style) saves ~15% wall-clock time per layer. Using this saved compute to add more layers (keeping total wall-clock time constant) yields better perplexity than the sequential baseline.  
**Falsification:** If extra layers don't compensate for the slight quality loss from parallel execution.

### Hypothesis 4: Micro-MoE Outperforms Dense at Any Scale
**Prediction:** Even at 50M parameters, a top-2-of-8 MoE with tiny experts (each ~1/4 the size of a standard FFN) will outperform a dense FFN because routing creates implicit specialization.  
**Falsification:** If routing entropy stays near maximum (all experts used equally) or if one expert dominates (routing collapse).

### Hypothesis 5: μP Enables Hyperparameter Transfer
**Prediction:** If we train with Maximal Update Parametrization (μP), the optimal learning rate, initialization scale, and other hyperparameters found at 50M will remain optimal at 500M without retuning.  
**Falsification:** If scaled-up model diverges or requires different hyperparameters despite μP.

---

## 6. Technical Implementation Details

### 6.1 Mixed Precision Training on T4

```python
# Key configuration for T4 GPUs
dtype = torch.float16  # T4 doesn't support bfloat16
scaler = torch.cuda.amp.GradScaler()  # Required for FP16 stability

# Critical: FP32 for these operations to prevent instability
# - Loss computation
# - Softmax in attention
# - LayerNorm/RMSNorm
# - Optimizer states (handled by AdamW automatically)
```

### 6.2 Multi-GPU Strategy (2× T4)

```python
# Option A: DataParallel (simpler, ~1.7× speedup)
model = torch.nn.DataParallel(model)

# Option B: DistributedDataParallel (better, ~1.9× speedup)
# Requires multiprocess launch but more efficient gradient sync
torch.distributed.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

Recommendation: Use DDP for production runs, DataParallel for quick ablations.

### 6.3 Memory Optimization

For a 50M model on 16GB T4:
- Model weights (FP16): ~100MB
- Optimizer states (FP32): ~400MB
- Activations (FP16, seq=512, batch=32): ~2–4GB
- Gradient accumulation headroom: ~2GB
- Total: ~4–7GB → Comfortable fit

If testing larger models (up to 125M on T4):
- Enable gradient checkpointing (recompute activations during backward pass)
- Use gradient accumulation (effective batch size = micro_batch × accumulation_steps)
- Flash Attention 2 (if available on T4) or memory-efficient attention from xformers

### 6.4 Efficient Data Loading

```python
# Use memory-mapped files for zero-copy data loading
# Pre-tokenize entire dataset to a flat binary file
# Each sample = contiguous block of token IDs

import numpy as np

class MemmapDataset:
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
```

### 6.5 Logging and Experiment Tracking

Since Kaggle has limited tools, use lightweight logging:

```python
# Log to CSV for easy post-analysis
# Columns: step, train_loss, val_loss, lr, tokens_per_sec, 
#           gpu_mem_mb, wall_time, grad_norm
import csv

# After training, save model + logs to Kaggle output
# Download and analyze locally
```

For local development, integrate Weights & Biases (wandb) if internet is available during initial setup.

---

## 7. Kaggle Notebook Execution Plan

### Notebook 1: Data Preparation (Run Once, Save to Dataset)

```
1. Download training corpus (OpenWebText subset, ~5GB compressed)
2. Train BPE tokenizer (sentencepiece, vocab=8192)
3. Tokenize entire corpus → save as .bin memmap file
4. Upload tokenized data + tokenizer as Kaggle Dataset
Runtime: ~1–2 hours
```

### Notebook 2: Ablation Runs (Multiple Sessions)

```
For each ablation:
1. Load pre-tokenized data from Kaggle Dataset
2. Configure model variant
3. Train for 200M tokens (~1 hour)
4. Evaluate validation perplexity
5. Log results
6. Save results to output
Runtime: 1–5 hours per session, multiple sessions
```

### Notebook 3: Full Training Run (Single Long Session)

```
1. Load data
2. Build NexusLM with best configuration
3. Train for 1–2B tokens (~8–10 hours)
4. Save checkpoints every 100M tokens
5. Final evaluation
Runtime: 10–11 hours (near full session)
```

### Notebook 4: Evaluation & Comparison

```
1. Load all checkpoints (baselines + Nexus)
2. Run full evaluation suite
3. Generate comparison tables and plots
4. Export results
Runtime: 1–2 hours
```

---

## 8. Expected Innovations Summary

| Component | Standard Approach | Our Innovation | Expected Benefit |
|-----------|------------------|----------------|-----------------|
| Attention | Multi-Head Attention | Differential Attention with multi-scale heads | Noise cancellation + multi-granularity |
| Position | Fixed RoPE | Learnable-frequency RoPE with per-layer gating | Better length generalization |
| Normalization | Pre-LN + RMSNorm | Sandwich-LN with QK-Norm + adaptive scaling | Training stability + quality |
| FFN | Dense SwiGLU | Sparse top-k SwiGLU or Micro-MoE | Better parameter utilization |
| Architecture | Sequential Attn→FFN | Parallel Attn+FFN with deeper network | Faster training, more depth |
| Residual | Uniform x + f(x) | Learned per-layer scaling α_l | Adaptive layer contribution |
| Training | AdamW + Cosine | SOAP/Muon + WSD schedule + batch ramp | Faster convergence |
| Init | Standard Xavier/He | μP parametrization | Scale-transfer of hyperparameters |
| Tokenizer | BPE 32K vocab | BPE 8K–16K vocab (smaller = more params for model) | Better param efficiency at small scale |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Innovations cancel each other out when combined | Test pairwise combinations before full assembly |
| Training instability with novel components | Always compare loss curves; abandon unstable variants early |
| Kaggle session timeouts | Save checkpoints frequently; design for resumable training |
| Overfitting on small data | Use dropout (0.1), weight decay (0.1), and monitor train/val gap |
| Evaluation noise at small scale | Run each ablation 2× with different seeds; report mean ± std |
| Innovation doesn't transfer to larger scale | Use μP to maximize transfer probability; validate in Phase 2 |
| GPU memory overflow on T4 | Profile memory before long runs; have fallback smaller configs |
| Tokenizer artifacts affecting comparisons | Use BPB (bits-per-byte) instead of perplexity for cross-tokenizer comparisons |

---

## 10. Phase 2 Preview — Scaling Up

**Trigger:** Phase 1 success criteria met (Section 4, Phase 1D).

**Phase 2 Plan (High-Level):**

1. **Scale to 350M parameters** on 4–8× A100/H100 GPUs
   - Train on 15–30B tokens (Chinchilla-optimal for this scale)
   - Full lm-evaluation-harness benchmark suite
   - Compare against Pythia-410M, OPT-350M, GPT-2 Medium

2. **Scale to 1.3B parameters** if 350M results hold
   - Train on 50–100B tokens
   - Compare against LLaMA-1.3B, Pythia-1.4B, TinyLlama-1.1B
   - Full MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA evaluation

3. **Publish findings** — whether positive or negative, the ablation results at small scale are valuable to the community

4. **Open-source** — Release model weights, training code, and all ablation results

---

## 11. Prompt Template for Guiding the AI Architect

When engaging the AI (Claude Opus 4.6 or similar) to execute this plan, use structured prompts like:

```
ROLE: You are a senior ML research scientist building a novel language model.

CONTEXT: We are in Phase 1B, Day 4 — Attention ablation.
Hardware: Kaggle 2×T4, 11h session limit.
Baseline: LLaMA-style 50M model, validation perplexity 28.3 at 200M tokens.

TASK: Implement Differential Attention as described in the Microsoft 
Diff Transformer paper. Integrate it into our NexusAttention module.
Train for 200M tokens on our prepared dataset.

REQUIREMENTS:
1. The implementation must be from scratch (no copying paper code)
2. Must work with mixed precision (FP16) on T4
3. Must log: step, train_loss, val_loss, tokens/sec, GPU memory
4. Compare final val perplexity against baseline (28.3)
5. If val perplexity < 27.0 (>5% improvement), mark as SUCCESS
6. Save all artifacts for later combination

OUTPUT: Complete Python implementation + training script + results analysis.
```

---

## 12. Success Metrics Dashboard

After all experiments, produce a summary table:

```
╔══════════════════════════════╦════════════╦═══════════╦═══════════════╦══════════════╗
║ Model                        ║ Val PPL    ║ Tok/sec   ║ Total FLOPS   ║ Peak Mem (GB)║
╠══════════════════════════════╬════════════╬═══════════╬═══════════════╬══════════════╣
║ GPT-2 Baseline (50M)        ║ 30.1       ║ 45,000    ║ 2.1e17        ║ 4.2          ║
║ LLaMA Baseline (50M)        ║ 28.3       ║ 42,000    ║ 2.1e17        ║ 4.5          ║
║ Nexus v1 (50M)              ║ ???        ║ ???       ║ ???           ║ ???          ║
╠══════════════════════════════╬════════════╬═══════════╬═══════════════╬══════════════╣
║ Target: ≥5% better PPL      ║ ≤26.9      ║           ║               ║              ║
║ OR: ≥20% fewer FLOPS        ║ ≤28.3      ║           ║ ≤1.68e17      ║              ║
╚══════════════════════════════╩════════════╩═══════════╩═══════════════╩══════════════╝
```

---

## 13. Long-Term Vision

If Phase 1 and Phase 2 validate our innovations:

1. **Architectural Template** — Package NexusLM as a configurable architecture that researchers can use as a drop-in replacement for standard Transformers.

2. **Scaling Laws Study** — Derive custom scaling laws for NexusLM architecture to predict optimal model size / data size / compute budget tradeoffs.

3. **Instruction Tuning** — Apply SFT (Supervised Fine-Tuning) and DPO/RLHF to create an instruction-following variant.

4. **Multimodal Extension** — If the architecture is more compute-efficient, the savings can be redirected toward vision encoders for a multimodal model.

5. **Community Release** — Fully open-source with training recipes, scaling studies, and pre-trained weights at multiple scales.

---

## Appendix A: Key Papers to Study

The AI architect should deeply study these before beginning implementation:

1. **Attention:** "Differential Transformer" (Microsoft, 2024), "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
2. **Architecture:** "PaLM: Scaling Language Modeling with Pathways" (Google, 2022) — for parallel attention+FFN, "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
3. **Training:** "Scaling Data-Constrained Language Models" (Muennighoff et al., 2023), "μP: Tensor Programs V" (Yang et al., 2022)
4. **Efficiency:** "Flash Attention 2" (Dao, 2023), "Mixture of Experts" (Shazeer et al., 2017; Fedus et al., 2022)
5. **Scaling Laws:** "Chinchilla: Training Compute-Optimal LLMs" (Hoffmann et al., 2022)
6. **Small Models:** "TinyStories" (Eldan & Li, 2023), "Pythia: A Suite for Analyzing LLMs" (Biderman et al., 2023)
7. **Normalization:** "DeepNet: Scaling Transformers to 1000 Layers" (Microsoft, 2022), "QK-Norm" (various, 2023–2024)
8. **Optimizers:** "SOAP: Improving and Stabilizing Shampoo using Adam" (Vyas et al., 2024)

---

## Appendix B: Checklist Before Each Kaggle Session

- [ ] Verify tokenized dataset is uploaded as Kaggle Dataset
- [ ] Confirm GPU quota is available (check Kaggle settings)
- [ ] Set notebook to GPU T4 × 2 accelerator
- [ ] Test data loading in first cell before training
- [ ] Set checkpoint save frequency (every 50M tokens minimum)
- [ ] Confirm output directory has enough space
- [ ] Set up auto-save of logs to Kaggle output
- [ ] Have fallback smaller config ready (in case of OOM)
- [ ] Timer: Set a 10-hour alarm (1 hour before Kaggle timeout)
- [ ] Previous run's results are saved and accessible

---

*This is a living document. Update after each experimental phase with findings, revised hypotheses, and adjusted plans.*
