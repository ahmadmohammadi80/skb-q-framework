# Reproducibility checklist

This document describes the reproducibility requirements for the code currently present in the `artifact/reproducibility-hf` branch. The branch is a **partial artifact**: it contains an executable Stage-3 sensitivity procedure, but not a verified end-to-end implementation of every method and experiment reported in the manuscript.

## Before generating results

- [ ] Record the exact base-model identifier and immutable revision/commit.
- [ ] Record the exact tokenizer identifier and immutable revision/commit.
- [ ] Record dataset name, configuration, split, and selection procedure.
- [ ] Record sample count, sequence length, preprocessing, padding, truncation, and special-token policy.
- [ ] Record candidate bit widths and quantization backend/granularity.
- [ ] Record the exact sensitivity definition and perturbation procedure.
- [ ] Record allocation objective and constraints when allocation is evaluated.
- [ ] Record random seeds and deterministic settings.
- [ ] Record hardware, Python, PyTorch, Transformers, CUDA, and relevant package versions.
- [ ] Preserve raw outputs and their configuration metadata.

## What the checked-in Stage-3 script actually does

`scripts/estimate_sensitivity_colab.py` performs a real calibration experiment for OPT-125M:

1. Load `facebook/opt-125m` and its tokenizer.
2. Load WikiText-2 raw-v1 training text.
3. Keep non-empty texts, shuffle them with the supplied seed, and select the requested number of samples.
4. Tokenize with `padding="max_length"`, truncation, and the supplied maximum length.
5. Evaluate FP calibration loss with padding positions masked from the loss.
6. Temporarily replace one eligible `torch.nn.Linear` weight at a time with a dequantized fake-quantized version.
7. Use symmetric per-tensor weight-only fake quantization for each requested candidate bit width.
8. Measure sensitivity as the increase in calibration loss relative to the FP baseline.
9. Restore the original weight after each perturbation and rank nodes by measured loss increase.
10. Write raw sensitivity rows and environment metadata to JSON.

This procedure is a sensitivity-estimation experiment. It is **not** the complete SKB-Q mixed-precision allocation pipeline.

## Determinism

The current script seeds Python, NumPy, and PyTorch. It does not explicitly call `torch.use_deterministic_algorithms(True)` and therefore the artifact should not claim strict deterministic execution. For a final publication-grade release, either enforce deterministic algorithms where supported or document the exact non-deterministic operations and their expected effect.

## Quantization semantics

The checked-in Stage-3 code uses:

- weight-only fake/simulated quantization;
- symmetric quantization;
- per-tensor scale;
- round-to-nearest via `torch.round`;
- dequantized floating-point weights returned to the model;
- no execution through hardware-specific low-bit kernels.

These semantics apply to the Stage-3 sensitivity script only. They must not automatically be attributed to the final manuscript experiments unless the corresponding final implementation is supplied and verified.

## Evaluation and evidence policy

Do not claim quality, memory, speed, or allocation improvements until the corresponding experiment has actually been executed and its raw output preserved. Keep raw measurements separate from summarized manuscript tables so that a reviewer can audit the reported values.

In particular, an existing script does not retroactively validate previously reported numerical results. Manuscript values must remain tied to the validated evidence package used for the final revision.

## Final-artifact requirements

Before calling the repository a complete reproducibility artifact, add and validate the missing end-to-end components for the manuscript's structural scoring, salience/full scoring, allocation, and evaluation procedures, and provide immutable model/tokenizer revisions plus a pinned environment.
