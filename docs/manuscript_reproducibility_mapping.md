# Manuscript ↔ GitHub reproducibility mapping

This note maps the audited GitHub artifact to the reproducibility-gap table in the SKB-Q manuscript.

## Recommended wording for the manuscript

### Row 15 — Quantization implementation / source code

**Recommended status:** Partially supplied; not a complete end-to-end implementation of the reported experiments.

**Recommended note:**

> The GitHub artifact contains executable Stage-3 sensitivity and quantization/allocation infrastructure on dedicated development branches, but the complete validated implementation used to generate every reported Structure, Salience, Full, Random, and final mixed-precision result is not established by the current public artifact. Therefore source-code availability should not be described as complete.

This is more accurate than either extreme:

- “Not supplied at all” — too strong because executable code exists.
- “Fully provided” — too strong because the audited public branches do not establish a one-to-one, end-to-end reproduction of all reported experiments.

### RTN / fake-quantization details

The `artifact/reproducibility-hf` Stage-3 script implements symmetric per-tensor weight-only fake quantization with round-to-nearest and dequantized floating-point weights. A separate development branch also contains a PyTorch fake-quantization backend, but its implementation is min-max/per-tensor and must not be silently conflated with the Stage-3 semantics or the final manuscript method.

### Allocation implementation

The development branch `cursor/quantization-allocation-layer-58b8` contains deterministic allocation infrastructure, including an allocation layer, bit-budget objects, candidate bit widths, constraints, and a deterministic resolver. This is evidence that allocation infrastructure exists, but it is not sufficient by itself to establish that the exact allocation procedure used for every manuscript table is reproducible from the public branch.

### Final experimental evidence

The manuscript's numerical results remain governed by the validated evidence package. Code availability and numerical evidence are separate claims. The existence of code should not be used to regenerate, replace, or “correct” reported values without a controlled rerun and evidence audit.

## Audit conclusion

The scientifically defensible repository statement is:

> **Partial reproducibility artifact; framework and selected executable quantization/sensitivity components are available, while complete end-to-end reproduction of all reported manuscript experiments is not yet established.**

This wording should be used consistently in the manuscript, appendix, README, and artifact metadata until a final release closes the remaining implementation-to-result mapping.
