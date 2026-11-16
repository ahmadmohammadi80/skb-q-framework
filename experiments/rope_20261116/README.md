# Real RoPE model experiment — 16 November 2026

Status: real-model controlled experiment completed; independent novelty and submission readiness NOT established.

Model: `HuggingFaceTB/SmolLM2-135M` at `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. WikiText-2 test: 595 blocks, 304,045 predicted tokens. All four quantized methods: **10,045,530 actual packed Q/K bytes**, including row scales and a 90-byte allocation map. Other model weights and activations remain FP32.

| Method | PPL |
|---|---:|
| FP32 anchor | 20.718691 |
| Identity + exact DP | 21.211038 |
| Fixed basis selected at 4/4 | 21.225418 |
| Precision-conditioned basis | 21.172071 |
| Wang2026 rule transplantation | 21.360729 |

The published comparator implements the angle estimator from Wang et al. (2026), Eq.3 / Appendix A.2.1, within our Q/K weight-only RTN and exact-allocation protocol. It is **not** the full published dynamic W4A4KV4/Hadamard pipeline. Do not label this row simply as a reproduction of the complete published method.

## Paired results

- Conditioned minus Fixed basis selected at 4/4: PPL delta -0.053347; exploratory circular moving-block bootstrap 95% interval [-0.068342, -0.037826], resampling length 8 blocks. See `summary.json` for lengths 1 and 16.
- Conditioned minus Identity + exact DP: PPL delta -0.038967; exploratory circular moving-block bootstrap 95% interval [-0.054360, -0.023590], resampling length 8 blocks. See `summary.json` for lengths 1 and 16.
- Conditioned minus Wang2026 rule transplantation: PPL delta -0.188658; exploratory circular moving-block bootstrap 95% interval [-0.223530, -0.153672], resampling length 8 blocks. See `summary.json` for lengths 1 and 16.

Fixed and conditioned maps differ in 12 bases and 0 precision choices. This experiment isolates basis choice, not a change in the final bit allocation.

## Reproduce and inspect

See [METHOD.md](METHOD.md) for the exact rule, proof scope, cost definition, comparator adaptations, references and commands. `run.py` records the 16-block pilot; `run_full.py` evaluates the complete test stream. `run_conditioned.py` is an equivalent single-mode entry point used to finish the conditioned evaluation concurrently. `compare_published.py` calibrates only on training text. `analyze.py` rejects incomplete evaluations and recomputes every reported PPL and paired interval.

`scores.json` is the actual model-derived option table. `verified_results.json` stores all block losses and maps. `decision_trace.json` identifies all changed groups. `packed_qk_manifest.json` records measured file sizes and hashes; `export_qk.py` regenerates each Q/K patch. Patches are not full-model checkpoints or packed inference kernels.

The full evaluation includes the pilot prefix. No selector setting was changed after inspecting that pilot. Only one checkpoint and one calibration prefix were tested. Intervals measure within-corpus variation, not independent replication. No speedup, whole-model six-bit precision, global precedence or acceptance claim is made.

## Relation to archived evidence

This experiment does not reuse the historical OPT archive results. The original archive and older prototype remain in the repository for provenance and must not be treated as validated evidence for the new rotation mechanism.
