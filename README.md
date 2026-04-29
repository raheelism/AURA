# AURA — NEXUS-AURORA Language Model

**AURA** is a research project implementing **NEXUS-AURORA**, a novel small-scale language model (~25–60M parameters) built from scratch. Rather than reusing a standard Transformer recipe, NEXUS-AURORA introduces a three-stream architecture — *Surface*, *Reasoning*, and *Verification* — where a private reasoning workspace runs in parallel with the token-generating stream and continuously grounds its outputs through a learned consistency gate.

The project is designed to be trainable entirely within the **Kaggle free tier** (2× T4 GPUs, ~11 h session) while demonstrating architectural improvements over a comparable LLaMA-style baseline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)  
2. [Repository Layout](#repository-layout)  
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
   - [Data Preparation](#data-preparation)  
   - [Training](#training)  
   - [Evaluation](#evaluation)  
   - [Ablation Studies](#ablation-studies)  
5. [Configuration Reference](#configuration-reference)  
6. [Model Components](#model-components)  
   - [Surface Stream](#surface-stream)  
   - [Reasoning Stream](#reasoning-stream)  
   - [Bridge Layer](#bridge-layer)  
   - [Verification Layer](#verification-layer)  
   - [Difficulty Estimator](#difficulty-estimator)  
   - [CoPE Positional Encoding](#cope-positional-encoding)  
7. [Training Details](#training-details)  
   - [Muon Optimizer](#muon-optimizer)  
   - [WSD Learning-Rate Schedule](#wsd-learning-rate-schedule)  
   - [Mixed Precision & Checkpointing](#mixed-precision--checkpointing)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [Baseline Model](#baseline-model)  
10. [Kaggle Workflow](#kaggle-workflow)  
11. [Running Tests](#running-tests)  
12. [License](#license)  

---

## Architecture Overview

NEXUS-AURORA processes every sequence through **three concurrent streams**:

| Stream | Dimension | Role |
|--------|-----------|------|
| **Surface (S)** | `d_surface = 512` | Token embeddings → causal LM head; the only stream that generates tokens |
| **Reasoning (R)** | `d_reasoning = 256` | 32 learned *slots*; private, non-causal workspace; never predicts tokens |
| **Verification (V)** | scalar gate | Measures S↔R consistency (surprise); gates how much R corrects S |

Each of the `n_blocks = 6` **AuroraBlocks** executes this pipeline in order:

```
Input S, R, gate_v
   │
   ├─ 1. PatternLayer   — causal local-window attention (window=64) + SwiGLU FFN
   ├─ 2. SemanticLayer  — full causal GQA (8 heads / 2 KV heads) + CoPE + SwiGLU FFN
   ├─ 3. DifficultyEstimator — predict K ∈ {1, 2, 4} reasoning iterations needed
   ├─ 4. ReasoningStream — R self-updates K times (non-causal slot attention)
   ├─ 5. BridgeLayer    — bidirectional S↔R cross-attention + gated sparse R→S write-back
   └─ 6. VerifyLayer    — compute surprise; emit gate_v for next block
```

**Key causality guarantee:** S is fully causally processed in steps 1–2 before any R information flows back. R reads from already-causal S states, so no future token leakage occurs through the reasoning workspace.

**Loss function:**

```
L = CrossEntropy(S) + λ_surprise · mean(surprise_losses)
```

where `λ_surprise = 0.1` pushes S and R toward mutual consistency over training.

---

## Repository Layout

```
AURA/
├── LICENSE
├── README.md
├── NEXUS_Implementation_Plan.md   # Full research spec (Phase 1 & 2 plan)
├── docs/
│   └── superpowers/
│       ├── plans/                 # Design plans
│       └── specs/                 # Component specs
└── nexus-lm/
    ├── requirements.txt
    ├── config/
    │   ├── nexus_aurora_v1.yaml   # Main NEXUS-AURORA config
    │   └── baseline_llama.yaml    # LLaMA baseline config
    ├── model/
    │   ├── model.py               # NexusAurora top-level model
    │   ├── block.py               # AuroraBlock (single layer)
    │   ├── surface.py             # PatternLayer, SemanticLayer, RMSNorm, SwiGLU
    │   ├── reasoning.py           # ReasoningStream (slot attention)
    │   ├── bridge.py              # BridgeLayer (S↔R cross-attention)
    │   ├── verify.py              # VerifyLayer (surprise gate)
    │   ├── difficulty.py          # DifficultyEstimator (adaptive K)
    │   ├── cope.py                # CoPE positional encoding
    │   └── baseline.py            # LLaMABaseline comparison model
    ├── data/
    │   ├── prepare.py             # Download & tokenize datasets
    │   ├── tokenizer.py           # SentencePiece BPE helpers
    │   └── dataloader.py          # MemmapDataset (zero-copy uint16)
    ├── training/
    │   ├── trainer.py             # Full training loop
    │   ├── muon.py                # Muon optimizer (Newton-Schulz ortho)
    │   └── scheduler.py           # WSD LR schedule
    ├── evaluation/
    │   ├── perplexity.py          # BPB (bits-per-byte) metric
    │   ├── routing_analysis.py    # K-distribution & surprise diagnostics
    │   └── ablation_runner.py     # Automated ablation suite
    ├── kaggle_scripts/
    │   ├── run.py                 # Unified CLI (data-prep / train / ablations / evaluate)
    │   └── README.md              # Kaggle-specific instructions
    ├── notebooks/
    │   ├── 01_data_prep.ipynb
    │   ├── 02_ablations.ipynb
    │   ├── 03_full_train.ipynb
    │   └── 04_evaluate.ipynb
    ├── results/                   # Checkpoints and CSV logs (gitignored)
    └── tests/                     # pytest test suite
```

---

## Installation

Python ≥ 3.10 and PyTorch ≥ 2.1 are required.

```bash
git clone https://github.com/raheelism/AURA.git
cd AURA/nexus-lm
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---------|---------|
| `torch>=2.1.0` | Core deep learning |
| `sentencepiece>=0.1.99` | BPE tokenizer |
| `datasets>=2.14.0` | HuggingFace dataset streaming |
| `numpy>=1.24.0` | Memory-mapped data arrays |
| `tqdm>=4.65.0` | Progress bars |
| `pyyaml>=6.0` | Config loading |
| `pytest>=7.4.0` | Test runner |
| `lm-eval>=0.4.0` | Standardized LM benchmarks |
| `wandb>=0.16.0` | Optional experiment tracking |

---

## Quick Start

All commands below assume you are in the `nexus-lm/` directory. Alternatively, use the unified `kaggle_scripts/run.py` CLI (see [Kaggle Workflow](#kaggle-workflow)).

### Data Preparation

Downloads FineWeb-Edu (1.5 B tokens, high-quality educational text) or TinyStories (100 M tokens, fast iteration), trains a BPE tokenizer (vocab size 8192), and writes memory-mapped binary files.

```bash
# FineWeb-Edu — full training run
python data/prepare.py \
  --output_dir data/ \
  --dataset fineweb_edu \
  --n_tokens 1500000000

# TinyStories — fast debugging run
python data/prepare.py \
  --output_dir data/ \
  --dataset tinystories \
  --n_tokens 100000000
```

Output files:

```
data/
├── tokenizer.model          # SentencePiece BPE model
├── fineweb_edu_train.bin    # uint16 memmap (~2.85 GB)
└── fineweb_edu_val.bin      # uint16 memmap (~29 MB)
```

### Training

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from data.dataloader import MemmapDataset
from model.model import NexusAurora, AuroraConfig
from training.trainer import Trainer, TrainerConfig

model = NexusAurora(AuroraConfig())
train_ds = MemmapDataset('data/fineweb_edu_train.bin', seq_len=512)
val_ds   = MemmapDataset('data/fineweb_edu_val.bin',   seq_len=512)
trainer  = Trainer(model, train_ds, val_ds, TrainerConfig(
    max_tokens=1_500_000_000,
    checkpoint_dir='results/checkpoints',
))
trainer.train()
"
```

Or use the Kaggle script:

```bash
python kaggle_scripts/run.py train \
  --train-bin data/fineweb_edu_train.bin \
  --val-bin   data/fineweb_edu_val.bin \
  --checkpoint-dir results/checkpoints \
  --max-tokens 1500000000
```

Training prints a progress line every 100 steps:

```
step=100 | loss=4.2831 | tokens=819,200 | lr=8.19e-06 | tok/s=12,432 | wall=66.0s
```

CSV logs are written to `results/checkpoints/train_log.csv` with columns:
`step, tokens_seen, train_loss, val_loss, lr, grad_norm, tokens_per_sec, wall_time`

**Resuming from checkpoint:**

```bash
python kaggle_scripts/run.py train \
  --train-bin data/fineweb_edu_train.bin \
  --val-bin   data/fineweb_edu_val.bin \
  --checkpoint-dir results/checkpoints \
  --resume-from results/checkpoints/ckpt_tokens_500M.pt \
  --max-tokens 1500000000
```

### Evaluation

```bash
python kaggle_scripts/run.py evaluate \
  --ckpt-path     results/checkpoints/ckpt_tokens_1500M.pt \
  --val-bin       data/fineweb_edu_val.bin \
  --tokenizer-path data/tokenizer.model \
  --output-json   results/eval_metrics.json
```

Outputs BPB, K-routing distribution, mean surprise, and a generated text sample. Results are saved as JSON:

```json
{
  "bpb": 1.2345,
  "k_1": 0.72,
  "k_2": 0.21,
  "k_4": 0.07,
  "surprise_mean": 0.043,
  "surprise_std": 0.018,
  "generated_text": "..."
}
```

### Ablation Studies

The ablation runner sweeps over 18 pre-defined configurations:

```bash
python kaggle_scripts/run.py ablations \
  --train-bin data/fineweb_edu_train.bin \
  --val-bin   data/fineweb_edu_val.bin \
  --ablations baseline aurora_full aurora_s_only aurora_s_r_k1 \
  --max-tokens 200000000
```

Available ablation names:

| Name | Description |
|------|-------------|
| `baseline` | LLaMA-style GQA + RoPE model (no reasoning stream) |
| `aurora_s_only` | Surface only — no R or V streams |
| `aurora_s_r_k1` | Surface + Reasoning with fixed K=1 |
| `aurora_s_r_adaptive` | Surface + Reasoning with adaptive K, no Verify |
| `aurora_s_v` | Surface + Verify gate only |
| `aurora_full` | Full NEXUS-AURORA (S + R + V, adaptive K) |
| `aurora_topk{1,4,8,16}` | Bridge top-k sparse write-back ablation |
| `aurora_kmax{2,4,6}` | Max reasoning iterations ablation |
| `aurora_slots{16,32,64}` | Reasoning slot count ablation |
| `aurora_{adamw,muon}` | Optimizer comparison |

Results are printed as a sorted table and written to CSV.

---

## Configuration Reference

### `config/nexus_aurora_v1.yaml`

```yaml
model:
  d_surface: 512          # Surface stream hidden dimension
  d_reasoning: 256        # Reasoning stream hidden dimension
  d_verify: 64            # Verification MLP hidden dimension
  n_reasoning_slots: 32   # Number of learned reasoning slot vectors
  n_blocks: 6             # Number of AuroraBlocks
  n_heads_surface: 8      # Query heads in SemanticLayer (GQA)
  n_kv_heads_surface: 2   # Key/Value heads in SemanticLayer
  n_heads_reasoning: 4    # Heads in ReasoningStream self-attention
  d_ffn_pattern: 512      # FFN width in PatternLayer
  d_ffn_semantic: 2048    # FFN width in SemanticLayer
  d_ffn_reasoning: 512    # FFN width in ReasoningStream
  local_window: 64        # PatternLayer causal window size (tokens)
  bridge_top_k: 8         # Top-K slots in sparse R→S write-back
  cope_positions: 16      # Number of CoPE learned position vectors
  vocab_size: 8192        # BPE vocabulary size
  max_seq_len: 512        # Maximum sequence length

training:
  batch_size: 16
  gradient_accumulation: 8          # Effective batch = 16 × 8 = 128 seqs
  max_lr: 1.0e-3
  min_lr: 1.0e-4
  warmup_tokens: 100_000_000        # ~100 M tokens linear LR warmup
  stable_tokens: 1_300_000_000      # ~1.3 B tokens at peak LR
  decay_tokens: 100_000_000         # ~100 M tokens cosine decay
  grad_clip: 1.0
  weight_decay: 0.1
  dtype: float16
  surprise_loss_weight: 0.1
  difficulty_entropy_weight: 0.01

optimizer:
  name: muon
  momentum: 0.95
  nesterov: true
  embed_lr: 3.0e-4       # AdamW LR for embeddings
  verify_lr: 1.0e-3      # AdamW LR for verification parameters

hardware:
  gpus: 2
  strategy: ddp
  gradient_checkpointing: true
```

### `config/baseline_llama.yaml`

LLaMA-style baseline matched to ~50M parameters for fair comparison:

```yaml
model:
  n_layers: 8
  d_model: 512
  n_heads: 8
  n_kv_heads: 2
  d_ffn: 1408
  vocab_size: 8192
  max_seq_len: 512
  rope_theta: 10000.0

optimizer:
  name: adamw
  betas: [0.9, 0.95]
```

---

## Model Components

### Surface Stream

The surface stream is responsible for all token-level processing and final next-token prediction.

**PatternLayer** (`model/surface.py`)  
Causal local-window self-attention (window = 64 tokens) + narrow SwiGLU FFN. Captures short-range syntactic patterns without paying full O(T²) cost.

**SemanticLayer** (`model/surface.py`)  
Full causal Grouped Query Attention (GQA, 8 query / 2 KV heads) with CoPE positional encoding + wide SwiGLU FFN. Captures long-range semantic dependencies.

Both layers use RMSNorm (pre-norm) and residual connections. Weight tying between the input embedding and the output LM head is enabled by default.

### Reasoning Stream

`model/reasoning.py` — a module containing 32 learned slot vectors (`nn.Parameter`), each of dimension `d_reasoning = 256`.

- **Non-causal**: all slots attend to all other slots (no causal mask).  
- **Never generates tokens**: exists only to improve surface stream quality.  
- **Adaptive depth**: runs K ∈ {1, 2, 4} self-attention + SwiGLU iterations per block, where K is determined dynamically by the DifficultyEstimator.

Slots are initialized from a zero-mean Gaussian (σ=0.02) and serve as learned concept primitives that are updated throughout the forward pass.

### Bridge Layer

`model/bridge.py` — bidirectional cross-attention between S and R.

Three operations per forward pass:

1. **S reads R (dense)**: each surface position attends to all 32 reasoning slots, importing contextual knowledge.  
2. **R reads S (dense)**: each reasoning slot attends to all surface positions, grounding itself in evidence.  
3. **Sparse R→S write-back (top-k=8)**: each surface position identifies its 8 most relevant reasoning slots and absorbs their values, **gated** by the verification gate `gate_v ∈ [0, 1]`.  

```
s = s + gate_v * sparse_r_to_s(s, r)
```

High surprise (gate_v → 1) allows strong R→S correction; low surprise (gate_v → 0) reduces the write-back to near-zero.

### Verification Layer

`model/verify.py` — computes a per-position *surprise* score measuring inconsistency between S and R.

```
r_pooled = mean(r, dim=slots)     # (B, T, d_r)
h = GELU(Linear([s; r_pooled]))   # (B, T, d_v)
surprise = sigmoid(Linear(h))     # (B, T, 1) ∈ [0, 1]
gate_v = 1 - surprise             # passed to BridgeLayer of next block
```

The surprise signal is also used as an auxiliary training loss (`L_surprise = mean(surprise)`). As training progresses, the model learns to make S and R mutually consistent, driving surprise toward zero.

### Difficulty Estimator

`model/difficulty.py` — a small two-layer MLP that predicts how many reasoning iterations each position requires.

- **Output**: K ∈ {1, 2, 4} per position.  
- **Training**: uses Gumbel-softmax (τ=1, hard=True) for gradient flow through the discrete choice.  
- **Inference**: argmax over logits.  
- **Batch efficiency**: the maximum K across all positions is used for the entire batch (GPU-friendly).  
- **Initialization**: biased toward K=1 at the start; the model learns to think harder as needed.

Entropy regularization (`difficulty_entropy_weight = 0.01`) encourages the model to use the full range of K values rather than collapsing to always K=1.

### CoPE Positional Encoding

`model/cope.py` — **Co**ntextual **P**ositional **E**ncoding, a content-dependent positional signal.

Unlike RoPE (fixed frequencies) or ALiBi (fixed biases), CoPE computes position as a **query-weighted sum** of `n_positions = 16` learned position vectors, all L2-normalized to the unit hypersphere:

```
gates = softmax(q @ pos_vecs.T / √d)   # (B, T, n_pos)
pos   = gates @ pos_vecs               # (B, T, d)
q_new = q + pos                        # inject into query
```

This means different queries (i.e., different content) can attend to positional information with different weights, enabling more flexible position-aware attention than fixed schemes.

---

## Training Details

### Muon Optimizer

`training/muon.py` — applies **Newton-Schulz orthogonalization** to gradient matrices before the weight update.

Only 2D linear weight matrices receive Muon updates. Embeddings, biases, and norm parameters use AdamW.

The key operation is 5 iterations of the NS5 polynomial approximation to the sign function, which computes the polar (orthogonal) factor of the gradient matrix:

```
X_{k+1} = (15/8)·X + (−5/4·A + 3/8·A²)·X,   A = X·Xᵀ
```

The orthogonalized gradient is then rescaled to match the RMS magnitude of the original gradient, ensuring that learning-rate semantics remain stable:

```python
p += -lr * (g_orth * rms_original / rms_orth)
```

This prevents weight matrix updates from collapsing or becoming redundant across the training run.

### WSD Learning-Rate Schedule

`training/scheduler.py` — **Warmup-Stable-Decay**, token-based.

| Phase | Tokens | LR Behavior |
|-------|--------|-------------|
| Warmup | 0 → 100 M | Linear ramp: 0 → `max_lr` |
| Stable | 100 M → 1.4 B | Constant at `max_lr = 1e-3` |
| Decay | 1.4 B → 1.5 B | Cosine: `max_lr` → `min_lr = 1e-4` |

The schedule is keyed on tokens seen (not steps), making it invariant to batch size and gradient accumulation changes.

### Mixed Precision & Checkpointing

- FP16 training via `torch.cuda.amp.GradScaler` on CUDA; float32 fallback on CPU/MPS.  
- Checkpoints saved every 50 M tokens (configurable) and on training completion.
- Each checkpoint contains model state, optimizer state, Muon state, scaler state, step count, tokens seen, and elapsed wall time (enabling accurate resumption across sessions).

---

## Evaluation Metrics

### Bits-per-Byte (BPB)

`evaluation/perplexity.py`

BPB is tokenizer-independent and enables fair cross-model comparison:

```
BPB = log₂(perplexity) / mean_bytes_per_token
```

For vocab=8192 on English text, `mean_bytes_per_token ≈ 3.5`. Lower BPB is better.

### Routing Analysis

`evaluation/routing_analysis.py`

Diagnostic metrics captured via forward hooks on all blocks:

| Metric | Description |
|--------|-------------|
| `k_1 / k_2 / k_4` | Fraction of positions routed to K=1, 2, or 4 reasoning iterations |
| `surprise_mean` | Mean per-position surprise score (lower = S and R more consistent) |
| `surprise_std` | Standard deviation of surprise scores |
| `total_positions_analyzed` | Total positions used for statistics |

These metrics reveal how the model allocates its reasoning compute and how well S and R have converged during training.

---

## Baseline Model

`model/baseline.py` — **LLaMABaseline**, a standard LLaMA-style decoder-only transformer for comparison.

Architecture: GQA + RoPE + SwiGLU + RMSNorm + weight-tied embedding. Configured to ~50M parameters to match NexusAurora's scale.

| Component | LLaMABaseline |
|-----------|---------------|
| Layers | 8 |
| Hidden dim | 512 |
| Attention | 8 heads / 2 KV heads (GQA) |
| Positional encoding | RoPE (θ=10000) |
| FFN | SwiGLU, d_ffn=1408 |
| Optimizer | AdamW |

---

## Kaggle Workflow

The `nexus-lm/kaggle_scripts/run.py` script is a unified CLI for running the entire pipeline from a single Kaggle notebook cell.

**1. Install dependencies**

```bash
!pip install -q -r /kaggle/working/nexus-lm/requirements.txt
```

**2. Data preparation**

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py data-prep \
  --dataset fineweb_edu \
  --output-dir /kaggle/working/data \
  --n-tokens 1500000000 \
  --verify
```

**3. Full training**

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py train \
  --train-bin /kaggle/input/nexus-data/fineweb_edu_train.bin \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --checkpoint-dir /kaggle/working/checkpoints \
  --max-tokens 1500000000
```

**4. Ablation run**

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py ablations \
  --train-bin /kaggle/input/nexus-data/fineweb_edu_train.bin \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --ablations baseline aurora_full aurora_s_only aurora_s_r_k1
```

**5. Evaluation**

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py evaluate \
  --ckpt-path /kaggle/working/checkpoints/ckpt_tokens_1500M.pt \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --tokenizer-path /kaggle/input/nexus-data/tokenizer.model \
  --output-json /kaggle/working/eval_metrics.json
```

Kaggle hardware constraints (Phase 1):

| Resource | Limit |
|----------|-------|
| GPU | 2× NVIDIA T4 (16 GB VRAM each) |
| System RAM | ~30 GB |
| Disk | ~70 GB |
| Session time | ~11 hours |

---

## Running Tests

All tests are in `nexus-lm/tests/` and use pytest. Tests cover every model component, the tokenizer, dataloader, optimizer, and trainer.

```bash
cd nexus-lm
pytest tests/ -v
```

Individual test files:

| File | Covers |
|------|--------|
| `test_model.py` | NexusAurora forward pass, generation |
| `test_block.py` | AuroraBlock end-to-end |
| `test_surface.py` | PatternLayer, SemanticLayer, RMSNorm, SwiGLU |
| `test_reasoning.py` | ReasoningStream slot attention |
| `test_bridge.py` | BridgeLayer cross-attention and gating |
| `test_verify.py` | VerifyLayer surprise computation |
| `test_difficulty.py` | DifficultyEstimator K routing |
| `test_cope.py` | CoPE positional encoding |
| `test_baseline.py` | LLaMABaseline forward pass |
| `test_trainer.py` | Trainer (single step, checkpoint round-trip) |
| `test_muon.py` | Muon optimizer, Newton-Schulz orthogonalization |
| `test_scheduler.py` | WSD schedule phases |
| `test_tokenizer.py` | BPE encode/decode |
| `test_dataloader.py` | MemmapDataset |

**Requirements for tests:** `torch`, `numpy`, `sentencepiece` must be installed (all included in `requirements.txt`).

---

## License

MIT License — Copyright © 2026 Muhammad Raheel Anwar. See [LICENSE](LICENSE) for full text.
