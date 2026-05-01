# NEXUS-AURORA: Causally Safe Private Latent Workspaces for Adaptive-Depth Language Modeling

## Abstract

Large language models usually apply a fixed amount of computation to every token, while reasoning-oriented inference methods often increase compute by generating additional visible tokens. We study whether an autoregressive language model can instead allocate variable computation inside a private latent workspace while preserving strict causal correctness. We introduce NEXUS-AURORA, a dual-stream architecture in which a surface stream predicts tokens and a prefix-local reasoning stream maintains private latent slots for each token position. A learned straight-through router selects recurrent depth from a small set of iteration counts, and a gated bridge writes latent workspace updates back to the surface stream. The current implementation includes full-model causality tests, differentiable routing and slot allocation, CE-only BPB evaluation, and executable ablations for reasoning, verification, fixed-depth, and adaptive-depth variants. The repaired system is ready for controlled re-evaluation; prior pre-repair BPB numbers should not be treated as publication evidence.

## 1. Motivation

Autoregressive LMs must satisfy a hard prefix-invariance constraint: changing future tokens must not change logits for earlier positions. This makes private latent reasoning harder than it first appears. A single global scratchpad can silently leak future information once it is reused across layers.

NEXUS-AURORA asks a narrower and testable question:

> Can a language model use a causally safe private latent workspace to improve quality or compute allocation relative to matched transformer baselines?

## 2. Architecture

### 2.1 Surface Stream

The surface stream is the only stream that produces logits. Each block applies:

1. A causal local-window pattern layer.
2. A full causal semantic attention layer.
3. Optional gated write-back from the private reasoning workspace.

### 2.2 Prefix-Local Reasoning Workspace

The repaired model uses reasoning slots with shape:

```text
(batch, sequence_position, reasoning_slot, reasoning_dim)
```

Each token position owns a private slot set. Position `t` can only ground its slots in surface positions `<= t`, so later tokens cannot contaminate earlier logits across blocks.

### 2.3 Adaptive Recurrent Depth

The difficulty estimator predicts a distribution over:

```text
K in {1, 2, 4}
```

During training, hard routing uses a straight-through Gumbel-Softmax estimator. The forward pass selects a discrete depth, while gradients flow through the relaxed router weights because those weights mix the recurrent-depth candidates. During evaluation, selected depths are executed per position.

### 2.4 Differentiable Slot Allocation

The slot allocator now uses straight-through top-k masking. The forward mask is sparse and hard, but gradients flow through a soft relaxation, allowing slot-allocation parameters to learn.

### 2.5 Verification Gate

The verification layer computes a per-position surprise score from the surface state and prefix-local reasoning state. The score contributes an auxiliary loss and gates subsequent reasoning write-back.

## 3. Causality Guarantee

The implementation now includes a full-model regression test:

```text
Changing tokens at positions >= t must not change logits at positions < t.
```

This test is applied to a multi-block model, which is where the previous global-workspace implementation failed.

## 4. Evaluation Plan

The next publishable experiment set should report:

1. CE-only validation loss and BPB.
2. Parameter-matched LLaMA-style baseline.
3. Surface-only AURORA.
4. Fixed-depth private workspace variants.
5. Adaptive-depth private workspace variants.
6. Verification-gate ablation.
7. Sparse vs dense bridge ablation.
8. FLOPs/token, tokens/sec, memory, and routing distribution.
9. Synthetic tasks where recurrent latent computation should matter.
10. At least three small-run seeds before any large claim.

## 5. Current Status

Implemented repair status:

- Full-model causality fixed with prefix-local reasoning slots.
- Difficulty router receives gradients.
- Slot allocator receives gradients.
- Later-block unused slot parameters removed from the trainable parameter set.
- BPB evaluation uses CE loss rather than auxiliary total loss.
- AURORA ablation switches now change model behavior.
- Baseline config updated to a roughly parameter-matched LLaMA-style baseline.

Not yet established:

- The repaired model has not been retrained.
- No post-repair BPB improvement is claimed.
- No NeurIPS-level empirical result is claimed yet.

## 6. Recommended Paper Claim After Re-Training

Use this form only if post-repair experiments support it:

> NEXUS-AURORA preserves autoregressive causality while enabling private latent recurrent computation. In controlled small-scale language-modeling experiments, adaptive private workspaces [improve/match] a parameter-matched baseline at [measured compute/quality tradeoff]. Ablations show that [component] accounts for [measured effect].

The paper should avoid claiming proof of reasoning, human-like cognition, or order-of-magnitude BPB gains unless those claims survive the repaired evaluation.
