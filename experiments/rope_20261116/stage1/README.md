# Independent architecture and stronger optimizer control

This follow-up executes the remaining empirical checks; it does **not** establish an independent new algorithm or submission readiness.

## Completed work

1. A standard alternating optimizer is compared with the exact joint finite-grid rule. It selects identical bit/basis maps at 5, 6 and 7 Q/K weight bits on SmolLM2 and Pythia (two iterations in all six cases).
2. Pinned Pythia-70M (GPT-NeoX with partial RoPE and biases) is evaluated on every complete 512-token WikiText-2 test block: 563 blocks, 287,693 predicted tokens. No test-based score/grid retuning is performed.
3. Fixed and precision-conditioned bases are evaluated at 5, 6 and 7 Q/K weight bits. Actual serialized patch sizes include row scales, FP32 biases and the precision map. This is not whole-model 5/6/7-bit quantization or integer-kernel inference.

| Q/K weight bits | Fixed basis PPL | Conditioned PPL | Difference | Actual patch bytes |
|---|---:|---:|---:|---:|
| 5 | 64.857700 | 64.831434 | -0.026266 | 2,015,280 |
| 6 | 61.947783 | 61.943992 | -0.003792 | 2,408,496 |
| 7 | 61.783265 | 61.779288 | -0.003978 | 2,801,712 |

FP32 reference: 61.380707. Identity basis at the same 6-bit budget: 61.904330, better than the conditioned basis. Confidence intervals are within-corpus circular block bootstraps, not independent training repetitions. See budget_summary.json and independent_summary.json.

## Interpretation

The finite-grid surrogate is solved correctly, but the stronger standard optimizer reproduces the solution. The held-out architecture does not reproduce the earlier benefit over the identity control at 6 bits. No claim of superiority over a full published rotation pipeline follows from a transplanted angle-rule control. The novelty gate remains open; these controls must be retained in any manuscript.

## Reproduction

Use the environment in the parent requirements.txt and pinned model in independent_model.json. From the repository root:

```bash
python experiments/rope_20261116/stage1/compare_joint.py
python experiments/rope_20261116/stage1/validate_solver.py
python experiments/rope_20261116/stage1/run_independent.py
python experiments/rope_20261116/stage1/run_budget_sweep.py
python experiments/rope_20261116/stage1/check_invariance.py
python experiments/rope_20261116/stage1/dataset_provenance.py
python experiments/rope_20261116/stage1/export_independent.py
python experiments/rope_20261116/stage1/analyze_independent.py
python experiments/rope_20261116/stage1/analyze_budget.py
```

Data and model weights require downloads. Reconstructed binary Q/K patches are not committed; their hashes, byte counts and decoding checks are recorded in independent_packed_manifest.json. No packed inference kernel is claimed.

## Numerical and provenance qualifications

- GPT-NeoX has a partial rotary subspace. Rotations leave nonrotary coordinates and V unchanged, and rotate Q/K biases along with weights.
- The score uses Q/K weights only; it omits bias-dependent error and assumes a weight-operator surrogate rather than measured activation covariance. The four offsets do not certify all contexts.
- FP32 invariance diagnostic: maximum logit discrepancy 0.03357 on logits of magnitude up to 1083.77; NLL change 0.0000324. FP64 eager-attention diagnostic: zero discrepancy at returned-output precision. This is not a universal numerical guarantee.
- The initial running script had an erroneous hardcoded theoretical byte metadata field (2,416,688). Its raw record is retained. The correct six-bit size, 2,408,496 bytes, is recomputed and checked by real serialization. The code now calculates this field; no model computation or reported NLL was changed by the correction.
- Score-table comparisons do not count as new forward passes. Identical maps imply identical quantized weights under the same deterministic implementation.

Proof scope is in PROOF_SCOPE.md; literature boundaries and the unresolved scientific gate are in LITERATURE_AND_CLAIMS.md. The main submission-blocking issue is independent contribution, not document formatting.
