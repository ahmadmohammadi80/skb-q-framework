from pathlib import Path
import json
p=Path(__file__).parent;s=json.load(open(p/'budget_summary.json'));r=json.load(open(p/'independent_summary.json'))
text='''# Independent architecture and stronger optimizer control

This follow-up executes the remaining empirical checks; it does **not** establish an independent new algorithm or submission readiness.

## Completed work

1. A standard alternating optimizer is compared with the exact joint finite-grid rule. It selects identical bit/basis maps at 5, 6 and 7 Q/K weight bits on SmolLM2 and Pythia (two iterations in all six cases).
2. Pinned Pythia-70M (GPT-NeoX with partial RoPE and biases) is evaluated on every complete 512-token WikiText-2 test block: 563 blocks, 287,693 predicted tokens. No test-based score/grid retuning is performed.
3. Fixed and precision-conditioned bases are evaluated at 5, 6 and 7 Q/K weight bits. Actual serialized patch sizes include row scales, FP32 biases and the precision map. This is not whole-model 5/6/7-bit quantization or integer-kernel inference.

| Q/K weight bits | Fixed basis PPL | Conditioned PPL | Difference | Actual patch bytes |
|---|---:|---:|---:|---:|
'''
for b in [5,6,7]:
 x=s[str(b)];text+=f"| {b} | {x['fixed_ppl']:.6f} | {x['conditioned_ppl']:.6f} | {x['delta_ppl']:+.6f} | {x['qk_bytes']:,} |\n"
text+=f"\nFP32 reference: {r['PPL']['fp32']:.6f}. Identity basis at the same 6-bit budget: {r['PPL']['identity']:.6f}, better than the conditioned basis. Confidence intervals are within-corpus circular block bootstraps, not independent training repetitions. See budget_summary.json and independent_summary.json.\n"
text+='''
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
'''
(p/'README.md').write_text(text)
root=(p/'root_readme_before.md').read_text();root+='\n## Stronger controls and independent architecture\n\n[Follow-up evidence](experiments/rope_20261116/stage1/README.md) adds an exact alternating-optimization control, Pythia-70M at three byte-matched budgets, full per-block losses, uncertainty intervals and serialized patch checks. The stronger optimizer reproduces the proposed choices. These findings do not resolve independent novelty or submission readiness.\n';(p/'root_readme_updated.md').write_text(root)
