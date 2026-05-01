# NEXUS-AURORA NeurIPS Readiness Report

Date: 2026-05-02

Scope reviewed:
- `NEXUS_AURORA_Paper_Draft.md`
- `NEXUS_Writeup.md`
- `nexus-lm/` implementation, tests, configs, and evaluation scripts
- Official NeurIPS 2026 call, deadlines, checklist, and review criteria

## Post-Repair Status

This report began as a pre-repair audit. The implementation issues identified below have now been addressed in code:

- Full multi-block causality now passes a regression test.
- The difficulty router receives gradients through straight-through Gumbel-Softmax routing weights.
- The slot allocator receives gradients through straight-through top-k masks.
- Trainable initial reasoning slots now live at the top-level model, removing unused per-block slot parameters.
- BPB evaluation now uses CE-only loss when available.
- Core AURORA ablation switches now change model behavior.
- The LLaMA baseline config now matches the roughly 50M-parameter baseline.

Current verification after repair:

```text
pytest -q
76 passed, 9 warnings

two-block causality probe:
max_past_logit_diff=0.0000000000
causal_ok True
```

The repaired system is still not a publication result until it is retrained and compared under the corrected evaluation pipeline.

## Executive Verdict

NEXUS-AURORA is interesting as a research direction, but it is not ready for a NeurIPS main-track submission as a results paper until the repaired implementation is retrained and evaluated.

The core idea has potential: a language model with a private latent reasoning stream, adaptive recurrence, and gated write-back could be publishable if it is made causally correct, if the adaptive-routing mechanism actually receives learning signal, and if the claims are rewritten around rigorous experiments rather than breakthrough language.

The original pre-repair audit found several publication-blocking issues:

1. The full multi-block model leaks future-token information through the global reasoning stream.
2. The claimed Gumbel-Softmax difficulty router is not actually trained by the loss.
3. The "token-level adaptive compute" claim is not true in the current forward pass; compute is batch-level/global per block.
4. Several ablation switches do not change the model, so core ablation claims cannot be trusted.
5. Reported BPB/loss numbers are internally inconsistent and not backed by local metrics artifacts.

My estimated NeurIPS outcome without new post-repair experiments: likely reject, because the corrected implementation does not yet have corrected empirical evidence. With clean ablations and a more honest framing, it could become a credible workshop paper or a future main-track "Concept & Feasibility" or "General" submission.

## NeurIPS 2026 Fit and Timing

Official NeurIPS 2026 sources say:
- Abstract deadline: May 4, 2026 AOE.
- Full paper deadline: May 6, 2026 AOE.
- Main track topics include language models, deep learning, optimization, theory, SysML infrastructure, and related areas.
- Review dimensions are Quality, Clarity, Significance, and Originality.
- The review form defines strong accept/accept as requiring technical solidity, strong evaluation, reproducibility/resources, and no unaddressed ethical concerns.
- NeurIPS 2026 asks authors to choose a contribution type: General, Theory, Use-Inspired, Concept & Feasibility, or Negative Results.

Given today's date in the local environment, May 2, 2026, the NeurIPS 2026 deadline is too close to repair the system, rerun experiments, and write a defensible paper unless the goal is a very high-risk submission. A better target is a NeurIPS workshop, COLM/ICLR/ICML next cycle, or NeurIPS 2027 main track after a rigorous rebuild.

Primary sources:
- NeurIPS 2026 Call for Papers: https://neurips.cc/Conferences/2026/CallForPapers
- NeurIPS 2026 Dates: https://nips.cc/Conferences/2026/Dates
- NeurIPS 2026 Main Track Handbook/review form: https://neurips.cc/Conferences/2026/MainTrackHandbook
- NeurIPS Checklist: https://nips.cc/public/guides/PaperChecklist
- Contribution types: https://blog.neurips.cc/2026/04/16/a-choice-of-contribution-types-at-neurips-2026/

## Original Code Audit Findings

### 1. Full Model Is Not Causal

The current bridge updates global reasoning slots from all surface positions, then later blocks let every earlier token read those slots. This leaks information from future tokens into past-token logits.

Relevant code:
- `nexus-lm/model/bridge.py:77` defines `_r_reads_s`.
- `nexus-lm/model/bridge.py:85` attends from reasoning slots to all surface positions.
- `nexus-lm/model/bridge.py:130` lets surface positions read reasoning slots.
- `nexus-lm/model/bridge.py:133` writes all surface positions into reasoning slots.
- `nexus-lm/model/model.py:119` carries the same reasoning state across blocks.

Local verification:

```text
pytest -q
71 passed, 8 warnings

custom two-block causality probe:
max_past_logit_diff=0.01073590
causal_ok False

custom one-block causality probe:
one_block_max_past_logit_diff=0.00000000
causal_ok True
```

This explains why the existing tests pass: the bridge is causal in a single block because the future-contaminated reasoning state is not reused until the next block. The leakage appears in the full architecture.

Impact: language-model validation loss and BPB can be invalid. A reviewer who checks causality would likely reject immediately.

Required fix:
- Make reasoning state position-indexed, prefix-indexed, or otherwise causally masked.
- Add a full-model causality regression test: changing tokens at positions `>= t` must not change logits at positions `< t`.
- Re-evaluate all results from scratch after the fix.

### 2. Difficulty Estimator Does Not Learn

The draft claims differentiable token-level adaptive halting via Gumbel-Softmax. The code samples K, converts it to an integer with `.item()`, and never uses `k_logits` in the loss.

Relevant code:
- `nexus-lm/model/difficulty.py:39` computes `F.gumbel_softmax`.
- `nexus-lm/model/difficulty.py:45` converts the max K to a Python int via `.item()`.
- `nexus-lm/model/difficulty.py:46` returns logits, but `AuroraBlock` never puts those logits into a loss.
- `nexus-lm/model/block.py:103` receives `k_logits`.
- `nexus-lm/model/block.py:116` uses only integer `k_batch`.
- `nexus-lm/model/model.py:30` defines `difficulty_entropy_weight`, but it is not used in the model loss.

Local gradient check:

```text
NO_GRAD blocks.0.difficulty.net.0.weight
NO_GRAD blocks.0.difficulty.net.0.bias
NO_GRAD blocks.0.difficulty.net.2.weight
NO_GRAD blocks.0.difficulty.net.2.bias
...
```

Impact: the routing distribution reported in `NEXUS_Writeup.md` is not evidence that the router learned meaningful difficulty. It may reflect initialization, drift from non-routing effects, or evaluation-time argmax behavior, but not a trained adaptive-halting mechanism.

Required fix:
- Either use a differentiable mixture over K paths, or implement a real straight-through estimator whose soft path influences the forward/loss.
- Add load-balancing, compute penalty, and entropy/anti-collapse terms if appropriate.
- Log per-layer router gradients and router loss terms.
- Report token-level compute actually saved.

### 3. Current Compute Is Not Token-Level Adaptive

The implementation chooses a single `k_batch` per block:

- `nexus-lm/model/difficulty.py:45`: `k_batch = int(k_tensor.max().item())`
- `nexus-lm/model/block.py:116`: `k_batch = max(int(k_batch_diff), int(halting_k))`
- `nexus-lm/model/block.py:118`: all slots run `k_batch - 1` more iterations.

This means the current model does not allocate different compute to different token positions during training. It computes per-token logits for diagnostics, but execution is batch/global per block.

Required paper-language correction:
- Do not claim "token-level adaptive compute" unless the executed graph really skips or adds computation per token.
- A fair claim might be: "a router predicts token-wise difficulty, but this prototype executes the maximum predicted K per batch for hardware simplicity." Then experiments must show either inference-time grouping or actual per-token sparse execution.

### 4. Ablation Runner Does Not Implement Core Ablations

The ablation config lists variants such as `aurora_s_only`, `aurora_s_r_k1`, `aurora_s_v`, and `aurora_full`, but the build path filters out `use_reasoning`, `use_verify`, and `fixed_k`, then returns the same full `NexusAurora` model.

Relevant code:
- `nexus-lm/evaluation/ablation_runner.py:29-33` defines S-only/S+R/S+V/full variants.
- `nexus-lm/evaluation/ablation_runner.py:66-67` excludes the keys that would change those variants.
- `nexus-lm/evaluation/ablation_runner.py:75` returns `NexusAurora(cfg)`.

Local check:

```text
aurora_s_only        NexusAurora 51759908 reasoning=True verify=True
aurora_s_r_k1        NexusAurora 51759908 reasoning=True verify=True
aurora_s_r_adaptive  NexusAurora 51759908 reasoning=True verify=True
aurora_s_v           NexusAurora 51759908 reasoning=True verify=True
aurora_full          NexusAurora 51759908 reasoning=True verify=True
aurora_kmax2         NexusAurora 51759908 reasoning=True verify=True
aurora_kmax6         NexusAurora 51759908 reasoning=True verify=True
```

Impact: the system currently cannot support claims that individual components help.

Required fix:
- Implement true model switches for no reasoning, no verification, fixed K, adaptive K, no bridge, causal bridge, dense vs sparse bridge, and different slot counts.
- Save every run's config, seed, git commit, checkpoint, log CSV, and metrics JSON.

### 5. Slot Allocator Is Hard Top-k and Receives No Gradient

Relevant code:
- `nexus-lm/model/slot_allocation.py:37` computes top-k indices.
- `nexus-lm/model/slot_allocation.py:39` scatters a hard binary mask.

Local gradient check shows all slot allocator parameters have `None` gradients.

Impact: dynamic slot allocation is not learned.

Required fix:
- Use a differentiable relaxation, straight-through hard concrete/top-k, Gumbel-top-k, or supervised/auxiliary signal.
- Or remove slot allocation from the claims.

### 6. Later Block Slot Parameters Are Unused

`NexusAurora.forward` initializes `r` only from `self.blocks[0].reasoning.slots`:

- `nexus-lm/model/model.py:111`

The slot parameters inside later blocks are never used and receive no gradient.

Local check:

```text
NO_GRAD blocks.1.reasoning.slots
NO_GRAD blocks.2.reasoning.slots
```

Impact: parameter accounting includes unused parameters; reviewers may see this as implementation immaturity.

Required fix:
- Move initial slots to the top-level model, or deliberately give each block its own reset/injection and document it.

### 7. BPB Calculation Is Not a Pure Language-Model Metric

`compute_bpb` calls `model(x, y)` and uses the returned `loss`:

- `nexus-lm/evaluation/perplexity.py:57`

For `NexusAurora`, that returned loss is not only cross-entropy:

- `nexus-lm/model/model.py:130` computes CE.
- `nexus-lm/model/model.py:135` adds the surprise auxiliary loss.
- `nexus-lm/model/model.py:137` returns total loss.

Impact: BPB for AURORA and baseline are not measured on exactly the same objective. BPB should be computed from token negative log-likelihood only.

Required fix:
- Return a metrics dict with `ce_loss`, `aux_loss`, and `total_loss`.
- Compute BPB from `ce_loss` only.
- Compute actual bytes/token from the tokenizer and eval corpus, not a fixed 3.5 constant, or justify the constant.

### 8. Baseline Parameter Comparison Is Confusing

`baseline_llama.yaml` uses `d_ffn: 1408`, which gives about 26.7M parameters. The test and ablation runner use `d_ffn=3300`, which gives about 50.0M parameters.

Local counts:

```text
Aurora default: 51,759,908 parameters
LLaMA d_ffn=1408: 26,747,392 parameters
LLaMA d_ffn=3300: 49,996,288 parameters
```

Impact: paper claims about "equivalent parameter counts" need a single canonical baseline config and saved run artifact.

Required fix:
- Use one baseline config file for the paper.
- Report parameters, active parameters, training FLOPs, inference FLOPs/token, throughput, and memory.

### 9. CoPE Implementation Is Not the Published CoPE Mechanism

The local `CoPE` implementation is a learned context-conditioned mixture of position vectors:

- `nexus-lm/model/cope.py:28-32`

Published Contextual Position Encoding uses context-dependent position increments/counting. If the paper calls this "CoPE", it needs either an exact implementation or a rename such as "contextual position-vector mixture."

## Claim Audit

Current risky claims in the draft/writeup:

1. "Order-of-magnitude decrease in BPB"
   - Not supported.
   - `NEXUS_AURORA_Paper_Draft.md:68-69` lists LLaMA BPB 32.97 and AURORA BPB 3.14.
   - A BPB of 32.97 would imply astronomical perplexity. If 32.97 is perplexity, then its BPB is about 1.44 with the current formula.

2. "Mathematically proving the Adaptive Halting hypothesis"
   - Not supported.
   - The router has no gradient and compute is not token-level adaptive.

3. "Without catastrophic routing collapse"
   - Not established.
   - A non-collapsed histogram is not enough; it needs trained router gradients, compute savings, and ablations.

4. "Full capabilities proven"
   - Not acceptable scientific language.
   - Replace with measured, narrow statements.

5. Result inconsistency:
   - `NEXUS_Writeup.md:11` says final loss 6.5364.
   - `NEXUS_Writeup.md:14` says BPB 2.6895, consistent with loss 6.5364 under the fixed 3.5 bytes/token formula.
   - `NEXUS_AURORA_Paper_Draft.md:73` says val loss 4.56 for the 250M-token run, which would imply BPB about 1.88, not 2.69.

Required rewrite posture:
- "We propose" rather than "we prove."
- "Prototype evidence" rather than "mathematical proof."
- "The current implementation uses max-K batching; future work implements token-sparse execution" unless fixed.
- "Causal-safe dual-stream latent recurrence" must be true before submission.

## Related Work Pressure

The novelty bar is real. NEXUS-AURORA must clearly distinguish itself from:

- Adaptive Computation Time (ACT), Graves 2016: https://arxiv.org/abs/1603.08983
- Universal Transformers with recurrent depth and adaptive computation, Dehghani et al. 2018/ICLR 2019: https://arxiv.org/abs/1807.03819
- Depth-Adaptive Transformer, Elbayad et al. 2019: https://arxiv.org/abs/1910.10073
- PonderNet, Banino et al. 2021: https://arxiv.org/abs/2107.05407
- Mixture-of-Depths, Raposo et al. 2024: https://arxiv.org/abs/2404.02258
- LayerSkip/early exit, Elhoushi et al. 2024: https://arxiv.org/abs/2404.16710
- Latent/continuous reasoning such as Coconut, Hao et al. 2024: https://arxiv.org/abs/2412.06769
- Recurrent-depth latent reasoning, Geiping et al. 2025: https://arxiv.org/abs/2502.05171
- Contextual Position Encoding, Golovneva et al. 2024: https://arxiv.org/abs/2405.18719

Potential novelty, if repaired:
- A causally safe private slot workspace for autoregressive LMs.
- Token or prefix-level latent recurrence with explicit compute accounting.
- A bridge/gate design that lets hidden reasoning improve token prediction without emitting thought tokens.
- Careful empirical analysis of when private latent slots help versus ordinary depth or MoD-style routing.

## Minimum Viable Paper After Fixes

### Core Research Question

"Can an autoregressive language model use a causally safe private latent workspace with adaptive recurrent depth to improve quality or compute efficiency relative to matched transformer baselines?"

### Non-negotiable Experiments

1. Causality validation
   - Full-model prefix invariance test.
   - Train/eval mode tested.
   - Multi-block tested.

2. Matched baselines
   - LLaMA-style 50M.
   - Surface-only typed architecture.
   - Same parameter budget.
   - Same token budget.
   - Same tokenizer.
   - Same data order/seed where possible.

3. Component ablations
   - S-only.
   - S + causal R with fixed K=1.
   - S + causal R with fixed K=2/4.
   - S + adaptive K.
   - S + R without verify gate.
   - S + R with verify gate.
   - Sparse bridge vs dense bridge.
   - Slots 8/16/32/64.

4. Compute accounting
   - Training tokens/sec.
   - Inference tokens/sec.
   - FLOPs/token theoretical and measured.
   - Active compute per token distribution.
   - Memory and KV-cache implications.

5. Language-model quality
   - Validation CE and BPB from CE only.
   - Perplexity.
   - At least 3 random seeds for small runs.
   - Larger single run if compute-constrained.

6. Behavioral tests
   - Short zero-shot benchmark suite if model size permits: HellaSwag, PIQA, ARC-Easy, WinoGrande.
   - Synthetic tasks where recurrence should matter: parity, nested parentheses, copying, algorithmic addition, needle-in-context.

7. Interpretability/diagnostics
   - Router distribution by token type and position.
   - Router gradients and entropy.
   - Slot utilization entropy.
   - Slot cosine diversity.
   - Bridge attention maps with causality-safe interpretation.

## Recommended Architecture Repair Paths

### Path A: Prefix-Indexed Reasoning State

Maintain `r_t` per position or per prefix. Each token position has reasoning slots that only read surface states `<= t`. This is easiest to make correct, but expensive: shape becomes `(B, T, N_slots, d_r)`.

Use it for a paper prototype and small-scale experiments. If it works, later optimize.

### Path B: Chunk-Causal Reasoning State

Maintain one reasoning state per chunk, where chunk `c` only reads earlier chunks and maybe the current chunk under causal masking. This is cheaper and publishable if compute savings are central, but it is less clean for token-level claims.

### Path C: MoD-Style Token Routing Instead of Global Slots

Use Mixture-of-Depths as the adaptive-compute backbone and add a private latent recurrent module only for routed tokens. This aligns better with hardware and existing literature, but novelty must be framed carefully.

### Path D: Recurrent-Depth LM Baseline With Private Slots

Start with a causal recurrent-depth transformer and add private slot memory as the delta. This gives a strong comparison against latent-depth reasoning work.

## Paper Positioning

Best paper type after fixes:
- Main Track, General, if experiments are strong and matched.
- Main Track, Concept & Feasibility, if the idea is novel and evidence is smaller but rigorous. The bar for this category is high; claims still need strong support.
- Workshop paper first if compute is limited.

Suggested title:
"Causally Safe Private Latent Workspaces for Adaptive-Depth Language Modeling"

Suggested thesis:
"We show that autoregressive language models can route selected positions through a private recurrent latent workspace without exposing intermediate thought tokens, while preserving strict causal correctness. In controlled small-scale experiments, the workspace improves quality/compute tradeoffs over matched transformer and adaptive-depth baselines."

Suggested contribution list:
1. A causally safe dual-stream architecture for private latent reasoning in autoregressive LMs.
2. A differentiable adaptive-depth router with explicit compute regularization.
3. A controlled ablation suite isolating private workspace, adaptive depth, verification gate, and bridge sparsity.
4. Diagnostics showing when and where latent compute is allocated.

## Proposed 8-Page NeurIPS Paper Outline

1. Introduction
   - Problem: LMs spend uniform compute per token and must externalize extra reasoning as text.
   - Gap: existing adaptive-depth methods do not provide a private causally safe workspace.
   - Contributions, stated modestly.

2. Related Work
   - Adaptive computation: ACT, Universal Transformers, PonderNet.
   - Transformer dynamic depth: Depth-Adaptive Transformer, Mixture-of-Depths, LayerSkip.
   - Latent reasoning: Coconut, recurrent-depth latent reasoning.
   - Position encoding and memory models.

3. Method
   - Surface stream.
   - Causal private workspace.
   - Adaptive recurrent depth.
   - Causal bridge and gated write-back.
   - Losses and compute regularization.

4. Causality and Compute Guarantees
   - Formal prefix-invariance statement.
   - Proof sketch or implementation invariant.
   - FLOPs/token calculation.

5. Experimental Setup
   - Data, tokenizer, model sizes.
   - Baselines and ablations.
   - Training budget.
   - Metrics.

6. Results
   - Main quality/compute table.
   - Ablation table.
   - Routing/slot diagnostics.
   - Generation and benchmark results if available.

7. Limitations
   - Small-scale compute.
   - Added implementation complexity.
   - Latent reasoning interpretability limits.
   - No claim of human-like reasoning.

8. Conclusion
   - What is established and what remains.

Appendix:
- Full configs.
- Causality tests.
- Additional seeds.
- Router/slot visualizations.
- NeurIPS checklist.

## Revised Abstract Template

Large language models typically apply a fixed computation budget to every token, while test-time reasoning methods often increase compute by generating additional visible tokens. We study whether an autoregressive language model can instead allocate variable compute inside a private latent workspace while preserving strict causal correctness. We introduce NEXUS-AURORA, a dual-stream architecture in which a surface stream predicts tokens and a causally masked recurrent workspace performs latent refinement for selected positions. A learned router controls recurrent depth under an explicit compute penalty, and a gated bridge writes workspace updates back to the surface stream. In controlled experiments at [X]M parameters and [Y]B training tokens, AURORA [matches/improves] a parameter-matched LLaMA-style baseline by [result], while using [compute result]. Ablations show that [private workspace/adaptive depth/gating] contributes [effect], and routing diagnostics indicate that additional latent compute is concentrated on [token types/tasks]. These results suggest that private latent workspaces are a promising route for compute-adaptive language modeling, while highlighting remaining challenges in scaling and interpretability.

## Concrete Next Steps

1. Fix full-model causality before any further training.
2. Fix the difficulty router so it receives gradient and controls real compute.
3. Remove or repair hard non-differentiable slot allocation.
4. Move slot initialization to a top-level parameter or use per-block slots intentionally.
5. Rewrite BPB evaluation to use CE-only NLL and actual bytes/token.
6. Implement true ablation switches.
7. Rerun small sanity experiments on synthetic tasks and TinyStories.
8. Rerun FineWeb-Edu only after causality and ablations pass.
9. Replace breakthrough/proof language with precise experimental claims.
10. Save all artifacts: configs, seeds, logs, checkpoints, metrics JSON, generated samples, and exact git commit.

## Bottom Line

The idea is not dead. In fact, the research direction is alive enough to be worth rescuing. But the current evidence cannot carry a NeurIPS paper. The fastest credible path is to turn this into a rigorous "causally safe private latent workspace" paper, with the first result being that the architecture is correct, measurable, and honestly compared. If the repaired model then beats or matches strong baselines at lower compute, the paper becomes much more compelling.
