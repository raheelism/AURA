# NEXUS-AURORA Repair Writeup

This file supersedes the pre-repair training notes. The earlier BPB and routing claims were produced before the causality and routing audit and should not be used as publication evidence.

## Repair Summary

1. Full-model causality fixed.
   - The reasoning workspace is now prefix-local with shape `(B, T, n_slots, d_reasoning)`.
   - Each position's slots can only read surface states from its own prefix.
   - A multi-block causality regression test now verifies that future-token edits do not alter past logits.

2. Difficulty routing made trainable.
   - The difficulty estimator now exposes straight-through Gumbel-Softmax routing weights.
   - Routing weights are used to mix recurrent-depth candidates, so gradients reach the router.
   - A routing balance auxiliary term is included through `difficulty_entropy_weight`.

3. Slot allocation made trainable.
   - The slot allocator now uses straight-through top-k masks.
   - Forward masks remain sparse and hard, but gradients flow through a soft relaxation.

4. Unused reasoning slot parameters removed.
   - Trainable initial reasoning slots now live at the top-level `NexusAurora` module.
   - `ReasoningStream.slots` is only a compatibility buffer for standalone use.

5. Evaluation fixed.
   - BPB now uses CE-only loss when a model returns metrics.
   - Auxiliary surprise/routing losses are no longer mixed into the language-model BPB metric.

6. Ablations fixed.
   - `use_reasoning`, `use_verify`, `fixed_k`, `max_k`, and slot-count overrides now change the actual AURORA model.
   - S-only, S+R fixed-K, S+R adaptive, S+V, and full AURORA variants are no longer all identical.

7. Baseline config fixed.
   - `baseline_llama.yaml` now uses `d_ffn: 3300`, matching the roughly 50M-parameter baseline used in tests and ablations.

## Verification

Current local test result after repair:

```text
76 passed, 9 warnings
```

Targeted probes after repair:

```text
full two-block causality probe: max_past_logit_diff=0.0000000000, causal_ok=True
router gradient: blocks.0.difficulty.net.0.weight grad present
slot allocator gradient: blocks.0.slot_allocator.context_proj.weight grad present
top-level reasoning_slots gradient present
```

## Publication Status

The repaired implementation is ready for new experiments. It is not yet a publication result.

Do not claim:

- order-of-magnitude BPB improvement,
- proof of adaptive halting,
- proven reasoning capabilities,
- post-repair superiority over baselines.

Safe current claim:

> The implementation now satisfies full-model autoregressive causality and includes trainable adaptive routing, differentiable slot allocation, CE-only BPB evaluation, and executable ablations. Controlled retraining is required before making performance claims.

## Next Experiment

Run a small controlled suite first:

1. LLaMA-style baseline, ~50M params.
2. AURORA S-only.
3. AURORA S+R fixed K=1.
4. AURORA S+R fixed K=2.
5. AURORA S+R adaptive K.
6. AURORA full with verification.

Report CE loss, BPB, tokens/sec, params, active mean K, routing distribution, and exact config for every run.