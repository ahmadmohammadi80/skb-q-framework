# Frozen diagnostic protocol and acceptance checks

## Before any measurement

1. Use Python 3.12 and the reviewed immutable Git commit with no tracked edits.
2. Install the pinned environment in an isolated venv. The launcher uses the CUDA
   12.4 PyTorch 2.6.0 wheel; CPU wheels are only for tests. A compatible NVIDIA
   driver is required. CUDA execution has not been verified by the CPU test suite.
3. Resolve the YAML's model, tokenizer, and dataset revisions once using
   `scripts/freeze_protocol.py --output frozen_protocol.json`. Archive that file.
   Resolution uses the Hub at preparation time, not an invented commit hash.
   Subsequent reproductions reuse this file; the resolver refuses to overwrite it.
4. Run tests, then the one-node smoke diagnostic, then the independent validator.
   A smoke result is marked `partial_scope: true` and cannot stand for a full run.
5. Inspect logs, configuration, samples, loss arithmetic, module scope, controls,
   and manifest before running all 72 block Linear modules.

## Exact measurement

- Model/tokenizer: OPT-125M at separately recorded immutable revisions.
- Data: Salesforce/wikitext, wikitext-2-raw-v1, train at a recorded revision.
- Strip whitespace, exclude empty rows, shuffle original row indices with Python
  Random(42), select 256 texts. Text duplicates are retained as dataset rows.
- Each text is separately tokenized with special tokens enabled, truncation to 512,
  and right padding to 512. These are not 256 full contiguous token blocks.
  Save row indices, texts, input IDs, attention masks, special-token mapping and
  dataset fingerprint; manifest hashes bind the complete tokenized inputs.
- Evaluate in FP16/eager attention, eval/inference mode, cache disabled.
  Compute cross entropy on FP32 logits, using logits[:, :-1] and labels[:, 1:].
  Ignore padding targets, accumulate per-sample NLL in FP64, then divide total
  NLL by the total number of valid prediction targets. Reject zero-target samples.
- Calibration perplexity is exp(calibration NLL), without clipping. Nonfinite
  quantities and overflow fail the run. This is not held-out benchmark perplexity.
- Scope: q/k/v/out projections and fc1/fc2 in the 12 transformer blocks (72 weights).
  Biases, embeddings and lm_head remain unchanged. Reject unexpected shared selected
  parameters; tied input/output embeddings are excluded from the perturbation scope.
- Quantizer: weight-only, per-tensor, symmetric narrow range [-qmax,+qmax],
  qmax=2^(bits-1)-1, absmax scale, round-to-nearest ties-to-even; bits 2/3/4/8.
  Scale, rounding and dequantization use FP32; then cast back to the model dtype.
  Handle zero tensors explicitly. This is fake quantization, not packed low-bit kernels.
- For every intervention start from the original weight and restore it in finally,
  including exception paths. Keep signed deltas, including negative/null results.
  Rank each bit width by descending delta, breaking ties by module name.

## Controls and reproducibility

Set Python/NumPy/PyTorch seeds, deterministic PyTorch algorithms, deterministic
cuBLAS workspace, disable cuDNN benchmarking and TF32, and fix eager attention.
Unsupported deterministic operations fail rather than silently fall back.
Record hardware, capability, CUDA runtime, full Python package versions and code SHA.
Deterministic settings do not establish cross-device bitwise equivalence.

The full run checks repeat baseline, batch size 1 versus configured batch size,
identity (16-bit no-op), and baseline recovery after each node. Frozen tolerances
are atol=1e-4 and rtol=1e-4; any failure requires investigation, not silent relaxation.
Tests also check perturbation order independence and exception restoration.
All selected weights are hashed before/after measurement; equality is required.

## Output and independent audit

Each execution uses an exclusive unique directory. JSON uses strict finite numbers
and atomic replacement. Interrupted/failed status cannot pass validation. Preserve
partial outputs and failures; never promote them or overwrite a previous run.

The manifest covers effective config, sample/token data, environment, measurements
and incremental rows. The validator checks hashes, run identity, completeness,
unique module/bit pairs, shifted target counts, per-sample sums, deltas, PPL,
rankings, restoration and numerical controls. It produces a separate report bound
to the manifest hash. It does not change the producer to `validated: true`.

Store raw per-sample NLLs for paired uncertainty analysis. A single calibration
sample/seed does not establish population-wide rank stability. If that claim is
needed, predefine additional sample selections and uncertainty analysis before
examining their results. Do not impose artificial monotonicity across bit widths.

`runtime_seconds_including_setup` covers setup after run directory creation and
all controls/interventions. It is not a low-bit inference speed benchmark.

## Manuscript gate

No verified legacy evidence package is supplied here. Any future claim needs an
exact evidence manifest and human/independent reproduction review. Full bit allocation,
parameter-weighted budgets, held-out evaluation and fair baselines are outside this
diagnostic. Training-calibration PPL must not be compared directly to legacy test
PPL or different sequence protocols. No novel algorithm, quality, speed, or memory
gain is established by successful diagnostic execution alone.
