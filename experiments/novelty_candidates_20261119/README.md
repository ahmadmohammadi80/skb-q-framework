# SKB-Q: finite-environment moment minimax allocation (2026-11-19)

**Research prototype; no general LLM, RoPE or independent-novelty claim.** Run `python skbq_multi_environment_allocator.py` in Python 3.10+; the test uses standard-library `fractions` only.

## Formal restricted score
For independent centered coordinate-independent X,Y, diagonal variance V, known per-coordinate fourth moments and a *fixed* two-token logit-error matrix E, compute exactly `S_e(E)=E[(X^T E X-X^T E Y)^2]` by second/fourth-moment contraction. This is a standard fourth-moment identity, **not a new theorem**.

Given a predeclared finite set of moment environments e, low/high error matrices for each module and integer bit upgrade costs, choose `b` minimizing `max_e sum_o S_e(E_(o,b_o))` subject to a byte budget. An exact Pareto-vector DP keeps nondominated accumulated score vectors at each cost. Proof: if a partial vector is coordinatewise dominated at identical cost, no common completion can improve its worst-environment score; induction yields the optimum. Pareto frontier size can be exponential. These proofs apply only to the finite additive model.

## Reproducible controls
- 60 exact rational Rademacher enumerations of the moment score.
- 250 random finite-budget allocation tests compared against exhaustive binary search.
- Toy 3-module one-upgrade witness: Gaussian-only bits (0,1,0), all-environment worst score 1041/16; minimax bits (0,0,1), worst score 129/4. **These are synthetic scores, not perplexity, runtime, packed bytes or a real-quantizer result.**

## Strongest novelty threats
- CASK-Q: already uses centered attention-error objective, paired Q/K and attention-KL bound: https://decentralizedcompute.company/cask-q
- NSDS: already calibration-free structure/numerical bit selection: https://aclanthology.org/2026.emnlp-main.1106/
- Standard multicriteria Pareto DP and fourth-moment identities are old.

**Missing:** measured Q/K quantizer error matrices; real RoPE/GQA causal context; actual model perplexity and equal packed storage with NSDS/CASK-Q; provenance of the moment uncertainty set. If that set is estimated using text, the method cannot be called calibration-free. Keep prior SKB-Q code unchanged.
