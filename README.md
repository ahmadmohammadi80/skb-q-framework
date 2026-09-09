# SKB-Q: Structural Knowledge-Based Quantization with Calibration-Optional Bit Allocation

Research artifact repository for the SKB-Q framework.

SKB-Q is a structural, auditable approach to mixed-precision quantization of transformer language models. This repository distinguishes **implemented research code**, **reproducibility configuration**, and **validated experimental evidence**. It does not treat placeholder, development, or partial pipeline outputs as final experimental evidence.

## Artifact status

**Current status: partial reproducibility artifact.**

The `artifact/reproducibility-hf` branch contains a real calibration-based, node-wise fake-quantization sensitivity script for OPT-125M plus configuration and reproducibility documentation. The checked-in script performs real forward passes and records calibration-loss deltas; it is not a placeholder sensitivity generator.

However, this branch does **not** currently contain a complete implementation of the full SKB-Q allocation/evaluation framework reported in the manuscript. In particular, the repository should not be interpreted as providing a complete, end-to-end implementation of every Structure, Salience, Full, Random, or final mixed-precision allocation experiment.

Accordingly, manuscript claims about source-code availability should describe this artifact as a **partial reproducibility artifact**, unless a later release adds and validates the missing end-to-end components.

## What is currently implemented

The checked-in Stage-3 script provides:

1. OPT-125M loading from Hugging Face.
2. WikiText-2 raw-v1 train calibration data selection.
3. Tokenization with padding/truncation to a configurable sequence length.
4. FP calibration-loss evaluation.
5. Temporary node-wise fake quantization of eligible `torch.nn.Linear` weights.
6. Symmetric **per-tensor**, weight-only fake quantization for candidate bit widths.
7. Calibration-loss delta as the measured sensitivity score.
8. Ranking of nodes separately for each candidate bit width.
9. JSON outputs containing raw rows and environment metadata.

The implementation is deliberately limited to sensitivity estimation. It does not by itself establish the final SKB-Q structural scoring/allocation results in the manuscript.

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

## Reproducibility protocol

Every validated experiment should record:

- exact base-model identifier and immutable revision/commit;
- tokenizer identifier and revision;
- dataset name, configuration, split, and selection procedure;
- number of calibration samples and sequence length;
- preprocessing, padding, truncation, and special-token policy;
- quantization backend and algorithm;
- candidate bit widths;
- sensitivity definition and perturbation procedure;
- allocation objective and constraints;
- random seeds and deterministic settings;
- software and hardware environment;
- evaluation metrics and raw outputs.

The current Stage-3 artifact records many of these fields, but not all are enforced by the script. See `docs/reproducibility.md` and `docs/artifact_audit.md`.

## Relationship to the manuscript

The manuscript reports a broader experimental framework than the code currently checked into this branch. The source artifact therefore supports **reproduction of the supplied Stage-3 sensitivity procedure**, but should not be cited as proof that the complete manuscript pipeline is available unless the missing allocation/evaluation implementation is subsequently added and validated.

The reported manuscript results must remain tied to their validated evidence package. No number should be regenerated, inferred, or replaced merely because a development script exists.

## Hugging Face artifact

The Hugging Face companion repository should contain data and machine-readable experimental artifacts. It should not be used to imply that source-code components absent from this GitHub repository are available.

## Citation

A `CITATION.cff` file is provided so the repository can be cited directly. Replace placeholder manuscript metadata with the final bibliographic information when available.

## License

The repository license applies to original code and documentation. Third-party model weights, datasets, and external assets remain subject to their original licenses and terms.
