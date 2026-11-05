# SKB-Q research artifact: calibration sensitivity diagnostic

**Partial reproducibility artifact. No CUDA experiment or manuscript result is certified by this repository.**

This branch provides an auditable OPT-125M diagnostic, not the complete structural
bit allocator. It measures paired changes in training-calibration NLL after
perturbing one transformer-block Linear weight. No sensitivity values, allocations,
quality gains, memory savings, or speedups are supplied as experimental results.

## Run in a fresh CUDA Colab runtime

Obtain `colab_stage3_opt125m.py` from the exact reviewed commit, then run it as normal
Python (not a script containing notebook magics):

```bash
python colab_stage3_opt125m.py --commit <40-character-reviewed-commit> --max-nodes 1
```

The launcher requires a new working directory, checks out the immutable commit,
creates an isolated Python environment, installs pinned dependencies, runs unit
tests, resolves immutable Hub revisions into a new frozen configuration, executes
a one-node smoke diagnostic, and runs the independent artifact validator. Every
subprocess is checked; failures stop execution. The default is deliberately partial.

Inspect and archive the entire run directory and logs. After reviewing the smoke
run, execute the full diagnostic from that same checkout and frozen configuration:

```bash
.venv/bin/python scripts/estimate_sensitivity_colab.py --config frozen_protocol.json --max-nodes 0
.venv/bin/python scripts/validate_sensitivity.py artifacts/sensitivity/<new-run-id>
```

Reproducing a previous run means reusing its `effective_config.json`, exact commit,
and environment; do **not** resolve the mutable Hub references again. The checked-in
YAML is a template and intentionally cannot run until all revisions are resolved.
`requirements-lock.txt` pins the resolved Python dependencies; the launcher selects
the PyTorch CUDA 12.4 wheel explicitly. See [the protocol](docs/reproducibility.md)
for hardware and reproducibility limitations.

## Evidence boundary

- `execution_status: completed` means the producer finished and its controls passed.
- `validation_report.json` reports independently recomputed artifact checks.
- `scientific_validation: pending` remains pending even when artifact checks pass.
- Hashes detect changes relative to the manifest; they do not prove that a person
  could not fabricate an entire run. Independent reruns and scientific review remain necessary.
- Synthetic unit-test fixtures are disposable test data, never manuscript evidence.

No externally validated evidence package is identified or verified in this branch.
Any future manuscript table must cite a specific immutable evidence manifest and
an independently inspected run. This implementation does not retroactively validate
legacy numbers. Quantization semantics changed during the audit: results produced
by older commits must remain separate.

## Scope and claims

The diagnostic uses calibration data. It is compatible with a data-free structural
allocator only as an external diagnostic of an allocation/prior frozen before
looking at these outcomes. Using these values to tune coefficients or choose bits
creates a calibration-dependent variant and must be labeled accordingly.

Per-node loss changes are protocol-dependent empirical measurements, not a proof
of additive joint sensitivity or novel bit-allocation performance. Future allocation
comparisons require held-out evaluation, matched parameter-weighted bit budgets,
matched quantizer semantics, fixed baselines, and raw allocation maps. The present
fake-quantized weights remain floating point; no deployment speed or storage claim
is supported.

## Files

- `scripts/estimate_sensitivity_colab.py`: diagnostic and producer controls.
- `scripts/freeze_protocol.py`: immutable revision resolution before measurement.
- `scripts/validate_sensitivity.py`: independent arithmetic/integrity checks.
- `tests/test_sensitivity.py`: CPU unit and tiny randomly initialized OPT controls.
- `docs/artifact_audit.md`: audit fixes, test scope, unresolved evidence boundary.

Third-party model and dataset licenses apply to those assets. Do not redistribute
raw calibration texts without checking the dataset's terms. Citation metadata in
`CITATION.cff` identifies software only, not a published or validated experiment.
