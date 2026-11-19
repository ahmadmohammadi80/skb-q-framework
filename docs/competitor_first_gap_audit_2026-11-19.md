# Competitor-first novelty audit — SKB-Q (2026-11-19)
Status: **literature/mechanism comparison only**. No new trained/quantized-model comparison run in this audit.

## Verified sources and overlap
| Prior work | What is already known | Specific novelty threat | Baseline role |
|---|---|---|---|
| NSDS (Zhang et al., EMNLP 2026) | **Calibration-free, structural AND numerical** sensitivity for mixed-precision layer bit allocation | "structural + data-free" is NOT sufficient as novelty | Strongest allocation-architecture competitor |
| CASK-Q (Decentralized Compute Company, Oct 2026) | Centered causal attention-logit objective, KL upper bound 1/4, joint Q/K quantizer calibration, exact empirical prefix-moment evaluator | **Do not claim centered objective, KL bound, or paired Q/K criterion as ours** | Nearest attention-error mathematical comparator; paper's experiments are synthetic, not full-LLM PPL |
| APTQ (Guan et al., DAC 2024) | Attention-aware mixed-precision quantization considering nonlinear downstream attention effects | "attention aware" and mixed precision already known | Strong calibration-based comparison, not identical input-information regime |

Primary URLs:
- https://aclanthology.org/2026.emnlp-main.1106/
- https://decentralizedcompute.company/cask-q
- https://arxiv.org/abs/2402.14866

## Candidate contribution in current prototype (NOT established novelty)
An exact **Gaussian-population** expectation of *previously known* centered causal attention-logit discrepancy under bias-free independent isotropic Gaussian inputs and fixed Q/K weights, with spectral RoPE all-offset contraction and a self-token fourth-moment correction, then an otherwise standard exact byte-budget allocator.

### What is demonstrably old / constrained
- Architecture structural bit scores alone (NSDS), attention-centered error and 1/4 KL bound (CASK-Q), mixed-precision allocation (APTQ), standard Gaussian fourth-moment identities, standard DP budget search.
- Real hidden states are not i.i.d. Gaussian. A recorded Rademacher counterexample shows equal covariance need not preserve Gaussian score ranking.
- In the recorded stage2 budget, self- and no-self variants select identical bit/basis maps. Therefore no empirical self-term contribution has been isolated.

## Missing direct comparisons (NOT yet executed)
1. On one pinned pretrained RoPE model and tokenizer, fixed Q/K quantizer and heldout tokens, evaluate the **same packed Q/K bytes** including metadata: random; per-layer structural; NSDS (faithful reproduction, if compatible); Gaussian full-offset score; no-self; four-offset score; and CASK-Q empirical criterion where matching training/calibration budget is disclosed.
2. Report candidate bit maps, PPL / paired per-block NLL, attention error, packed storage, allocation-search CPU time, calibration/GPU time, inference latency only if kernel changes.
3. CASK-Q vs calibration-free comparison must show input-information and compute-budget inequalities explicitly; never call a candidate superior merely because CASK-Q has not reported real-model PPL.
4. For self-term identification, construct/choose a **frozen, preregistered** budget and architecture where self and no-self bit maps differ; do not tune on the final test corpus.
5. Verify RoPE frequency sign, attention normalization, GQA grouping and identical token counts before interpreting any performance difference.

## Gate
No independent novelty claim until a nontrivial score/optimization distinction versus CASK-Q and NSDS and a matched evidence chain are documented; do not claim unmeasured PPL or universal Gaussian robustness. Comparison audit is not a benchmark.
