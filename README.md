# SKB-Q: Structural Knowledge-Based Quantization with Calibration-Optional Bit Allocation

Public research artifact repository for the SKB-Q manuscript.

> **Artifact status: Partial reproducibility artifact**
>
> This repository provides the available implementation components, reproducibility configuration, and documentation. It does **not** claim to be a complete end-to-end replication package for every experiment or numerical result reported in the manuscript.

## What this repository is for

SKB-Q is a structural, auditable approach to mixed-precision quantization of transformer language models. This branch is organized so that a reviewer can distinguish:

- **implemented research code**;
- **reproducibility configuration**; and
- **validated experimental evidence**, which is maintained separately from the executable code.

The repository deliberately does not treat placeholder, development, or partial-pipeline outputs as final experimental evidence.

## Reviewer quick start

1. Read [`docs/artifact_audit.md`](docs/artifact_audit.md) for the exact evidence boundary.
2. Read [`docs/reproducibility.md`](docs/reproducibility.md) for the execution and reporting checklist.
3. Inspect [`configs/opt125m_calibration.yaml`](configs/opt125m_calibration.yaml) for the supplied Stage-3 configuration.
4. Run [`scripts/estimate_sensitivity_colab.py`](scripts/estimate_sensitivity_colab.py) to reproduce the supplied sensitivity-estimation procedure, subject to the documented environment and revision limitations.

## What is currently implemented

The checked-in Stage-3 script provides:

1. OPT-125M loading from Hugging Face.
2. WikiText-2 raw-v1 training-text calibration data.
3. Configurable tokenization, padding, and truncation.
4. Floating-point calibration-loss evaluation.
5. Temporary node-wise fake quantization of eligible `torch.nn.Linear` weights.
6. Symmetric **per-tensor**, weight-only fake quantization for candidate bit widths.
7. Calibration-loss delta as the measured sensitivity score.
8. Separate ranking of nodes for each candidate bit width.
9. JSON output containing sensitivity rows and environment metadata.

This is a **sensitivity-estimation procedure**, not the complete SKB-Q mixed-precision allocation/evaluation pipeline used to establish all manuscript results.

## Repository structure

```text
skb-q-framework/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
│   └── opt125m_calibration.yaml
├── docs/
│   ├── reproducibility.md
│   └── artifact_audit.md
└── scripts/
    └── estimate_sensitivity_colab.py
```

## Reproducibility boundary

The supplied code supports reproduction of the checked-in Stage-3 sensitivity procedure. It does **not** establish reproduction of every Structure, Salience, Full, Random, or final mixed-precision allocation experiment reported in the manuscript.

In particular:

- the supplied Stage-3 implementation uses symmetric per-tensor fake quantization;
- the final manuscript experiments must not be assumed to use identical quantization semantics solely because this script exists;
- model and tokenizer revisions in the example configuration use mutable `main` references and therefore do not constitute immutable publication-run identifiers; and
- numerical manuscript results remain tied to the validated evidence package rather than being regenerated from a development script.

These boundaries are intentional and are documented for reproducibility transparency.

## Reproducibility protocol

Every validated experiment should record:

- exact base-model identifier and immutable revision/commit;
- tokenizer identifier and immutable revision;
- dataset name, configuration, split, and selection procedure;
- number of calibration samples and sequence length;
- preprocessing, padding, truncation, and special-token policy;
- quantization backend and algorithm;
- candidate bit widths;
- sensitivity definition and perturbation procedure;
- allocation objective and constraints;
- random seeds and deterministic settings;
- software and hardware environment; and
- evaluation metrics and raw outputs.

The current Stage-3 artifact records many of these fields, but not all are enforced by the script. See [`docs/reproducibility.md`](docs/reproducibility.md) and [`docs/artifact_audit.md`](docs/artifact_audit.md).

## Relationship to the manuscript

The manuscript reports a broader experimental framework than the code currently checked into this public artifact branch. Accordingly, the repository should be cited as a **partial reproducibility artifact** rather than a complete replication package.

The validated experimental evidence package remains the authoritative source for the reported numerical results. No reported value should be inferred, substituted, or regenerated merely because a development implementation is present in this repository.

## Hugging Face evidence artifact

The companion Hugging Face evidence repository contains machine-readable experimental artifacts supporting the reported results. It should not be interpreted as supplying source-code components that are absent from this GitHub repository.

## Citation

A `CITATION.cff` file is provided so this repository can be cited directly. Replace placeholder manuscript metadata with final bibliographic information when available.

## License

The repository license applies to original code and documentation. Third-party model weights, datasets, and external assets remain subject to their original licenses and terms.
