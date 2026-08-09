# Reproducibility checklist

## Before generating results

- [ ] Record the exact base-model identifier and revision.
- [ ] Record tokenizer identifier and revision.
- [ ] Record dataset name, configuration, split, and version/revision when available.
- [ ] Record sample count, sequence length, preprocessing, and seed.
- [ ] Record candidate bit widths and quantization backend.
- [ ] Record the exact sensitivity definition.
- [ ] Record hardware, Python version, PyTorch version, Transformers version, and CUDA version.

## Sensitivity estimation

Sensitivity must be computed from an actual calibration procedure. A structural proxy or placeholder value may be useful during development, but it must not be reported as a measured sensitivity result.

A recommended first implementation is node-wise perturbation under a fixed calibration set:

1. Evaluate the FP baseline loss on the same calibration samples.
2. Perturb one eligible operator/node at a time using each candidate bit width.
3. Re-evaluate the calibration loss under the same samples and deterministic settings.
4. Store the loss delta and all metadata needed to reproduce it.
5. Aggregate/normalize node-level sensitivity only after raw measurements are preserved.

The output should make the provenance explicit, for example:

```json
{
  "model": "facebook/opt-125m",
  "dataset": "wikitext/wikitext-2-raw-v1",
  "split": "train",
  "samples": 256,
  "max_length": 512,
  "seed": 42,
  "method": "calibration_loss_delta",
  "candidate_bits": [2, 3, 4, 8],
  "validated": true
}
```

## Evaluation

Do not claim quality, memory, speed, or allocation improvements until the corresponding experiment has actually been executed. Keep raw measurements separate from summarized tables so that a reviewer can audit the reported values.
