# SKB-Q artifact audit

Audit date: 2026-09-09
Branch audited: `artifact/reproducibility-hf`

## Executive finding

This branch contains a **real but partial reproducibility artifact**. It contains an executable OPT-125M calibration/sensitivity procedure based on real forward passes and temporary node-wise fake quantization. It does **not** contain a verified end-to-end implementation of the complete manuscript pipeline for structural scoring, salience scoring, full scoring, bit allocation, and final PPL evaluation.

Therefore:

- The repository may be cited as a partial source-code/reproducibility artifact.
- The Stage-3 sensitivity script may be used to reproduce that sensitivity procedure after environment/configuration details are fixed.
- The repository must not be described as a complete implementation of all manuscript experiments.
- Existing manuscript numerical results must remain anchored to the validated evidence package; the presence of this development/reproducibility script does not validate or replace those results.

## Code audit

### Verified in the branch

` scripts/estimate_sensitivity_colab.py ` implements:

- Hugging Face OPT-125M loading.
- WikiText-2 raw-v1 train loading.
- Selection of non-empty training texts followed by seeded shuffling.
- Tokenization with `padding="max_length"`, truncation, and configurable `max_length`.
- Padding-aware causal-LM labels using `-100`.
- FP calibration-loss evaluation over the same calibration set.
- Temporary quantization of individual `torch.nn.Linear` module weights.
- Symmetric per-tensor weight-only fake quantization.
- Candidate bit widths configurable from the command line.
- Sensitivity defined as quantized calibration loss minus FP baseline loss.
- Per-bit sensitivity ranking.
- JSON output containing raw measurements and runtime metadata.

### Not verified as implemented in this branch

The audited branch does not expose a complete source implementation for:

- the manuscript's structural scoring function;
- the manuscript's salience scoring function;
- the manuscript's Full scoring combination;
- the final mixed-precision bit-allocation solver/policy;
- the manuscript's full allocation/evaluation pipeline across all reported methods;
- hardware-executed low-bit kernels;
- an end-to-end reproduction script for the reported OPT-125M/OPT-1.3B/Mistral results.

The README previously described a broader planned repository structure than the files actually present. This audit corrects that documentation rather than claiming absent code exists.

## Important implementation mismatch to preserve in the manuscript audit

The checked-in sensitivity script uses **symmetric per-tensor** fake quantization. It must not be used as evidence for a manuscript statement that the final reported SKB-Q experiments use a different granularity (for example, per-row) unless the corresponding implementation is separately supplied and verified.

Likewise, the script's sensitivity experiment is calibration-loss-based and node-wise. It is not itself the final structural, salience, or full allocation method.

## Reproducibility risks found

1. The configuration uses `revision: main`, which is mutable and therefore not an immutable model revision.
2. The script seeds Python, NumPy, and PyTorch, but does not explicitly enable PyTorch deterministic algorithms. A configuration flag claiming `deterministic: true` would therefore overstate what the script enforces.
3. The tokenizer/model revision and exact software versions are not all pinned by the launcher.
4. The launcher upgrades packages at runtime rather than installing a fully pinned environment.
5. The script records environment versions in output, but the repository does not contain the generated validated sensitivity outputs themselves.
6. The configured 256 calibration samples are a Stage-3 sensitivity configuration and should not be confused with the manuscript's separate Mistral 300-text ablation protocol.

## Required manuscript treatment

For the reproducibility-gap table, the appropriate interpretation is:

- **Quantization implementation/source code:** partially supplied; the repository contains the Stage-3 node-wise fake-quantization sensitivity implementation, but not the complete manuscript pipeline.
- **RTN/fake quantization:** specified for this Stage-3 artifact as symmetric per-tensor weight-only fake quantization.
- **Final allocation implementation:** not supplied/verified in this branch.
- **Complete end-to-end source artifact:** not supplied/verified.

This distinction prevents both under-reporting (ignoring the real code that exists) and over-claiming (calling the partial scaffold a complete implementation).
