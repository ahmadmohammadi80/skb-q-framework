# SKB-Q causal-context score candidate

This checkpoint adds an executable scoring mechanism and its proof under a restricted input model. It does not establish independent novelty, a new DP, downstream superiority, or submission readiness.

## New executable output

`context_score.py` integrates the expected row-centered causal attention-logit error over all context offsets under independent standard Gaussian token inputs and bias-free Q/K projections. It includes the correct same-token fourth-moment term and a reusable RoPE spectral kernel. `run_context.py` connects its option table to the existing exact byte-budget allocator. The four rotation angles and quantizer are unchanged.

The score upper-bounds average expected attention-row KL **under the specified Gaussian model only**. It does not certify real LLM perplexity or real hidden-state distributions. See `PROOF.md` for the derivation and assumptions.

## Completed checks

- 32 comparisons with explicit finite-context evaluation: maximum relative discrepancy 2.55e-14.
- 120,000 Gaussian sequences: analytic expectation agrees with Monte Carlo within the recorded sampling uncertainty; the sampled attention KL does not exceed the samplewise centered-error bound.
- Unmodified official CASK-Q prefix code at commit `9b1eaac51ce1ba223ffedbe7aef1dd3ff5bb69a2`: dense/prefix values agree; its Gaussian population expectation agrees with the new analytic evaluator after normalization. This is a core-objective comparison, not reproduction of full CASK-Q quantizer training.
- 12 tests in the official reference repository passed in this environment.
- A deterministic quantizer-generated example changes the selected equal-budget precision pair from Q4/K8 to Q8/K4. The matrices and every evaluated score are recorded in `separation_example.json`; this is an existence witness, not a general performance benchmark.
- On a single-thread CPU prototype, d=64, input width=576, context=512, the new contraction takes median 1.526 ms versus 48.641 ms for explicit Gram traversal over all offsets (7 repetitions). Setup costs are separately recorded. This approximately 31.9x ratio concerns the same synthetic score computation, not model inference or empirical CASK-Q calibration.

## Closest prior work and claim boundary

CASK-Q already supplies the centered causal error objective, its sharp 1/4 attention-KL bound, and an exact empirical prefix-moment evaluator. Those components are not new here. The candidate contribution is an analytic, calibration-free Gaussian population evaluator using the full RoPE context and self-token correction. Spectral averaging and Gaussian moment identities are standard ingredients; the independent contribution of their present use remains to be evaluated.

Primary sources:
- https://decentralizedcompute.company/cask-q
- https://github.com/the-decentralized-compute-company/quantization
- APTQ, attention-aware mixed precision: https://arxiv.org/abs/2402.14866

## Real-model status at this checkpoint

All candidate score tables have been computed for the pinned SmolLM2-135M checkpoint. `progress_snapshot.json` records only completed forward evaluations at the time of the checkpoint. The negative-offset four-position control is a control, not the proposed full-context score. No unfinished evaluation is promoted into a measured PPL result.

The self-term and no-self variants select identical bit/basis maps at this budget, so this real-model setting cannot isolate the self-term mechanism. Equal selected maps imply equal quantized weights under the same implementation. This limitation must remain in any report, even if the eventual PPL is favorable.

## Reproduction

Use the parent environment. Run:

```bash
OPENBLAS_NUM_THREADS=1 python experiments/rope_20261116/stage2/validate_context.py
OPENBLAS_NUM_THREADS=1 python experiments/rope_20261116/stage2/benchmark_kernel.py
OPENBLAS_NUM_THREADS=1 python experiments/rope_20261116/stage2/find_example.py
OPENBLAS_NUM_THREADS=1 python experiments/rope_20261116/stage2/run_context.py
```

For the official-core comparison, clone the reference repository, check out the exact commit above, and pass its local directory to `compare_cask.py`. No third-party source is vendored in this checkpoint. The full model evaluation is exploratory: this corpus was already inspected in previous stages. No angles, scoring settings or budgets were tuned after locking `protocol.json`.
