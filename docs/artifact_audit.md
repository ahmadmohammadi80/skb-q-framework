# Pre-execution audit fixes

The former script's automatic `validated: true`, unshifted loss weighting, capped
perplexity, implicit module scope, mutable loaders, unused YAML, FP16 quantizer
arithmetic, unchecked restoration, notebook-magics launcher and overwrite-prone
outputs have been replaced.

The current artifact provides a consumed frozen configuration, explicit 72-module
scope excluding the tied head, FP32 quantizer arithmetic, paired per-sample NLL,
strict JSON and unique run directories, restoration controls, and independent
artifact validation. Documentation no longer asserts that an unspecified external
evidence package has already been validated.

Validation scope: CPU tests use synthetic tensors and small randomly initialized
OPT configurations without downloading pretrained weights or WikiText. They check
loss against the Transformers implementation, padding/shift accounting, batch and
perturbation-order invariance, identity controls, zero/tiny weights, finite outputs,
shared-parameter rejection, restoration on exceptions, and malformed artifacts.
These tests are software evidence only. See `validation/cpu-tests.txt` for the
recorded test run and environment; no CUDA result is implied.

Remaining execution gates: actual compatible CUDA environment, immutable Hub assets,
one-node smoke execution, artifact inspection, then the full diagnostic. The
allocation algorithm, held-out evaluation and manuscript numerical claims remain
outside this artifact. Scientific validation stays pending after automated checks.
