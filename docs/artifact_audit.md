# SKB-Q Artifact Audit

## Purpose

This repository is the public code and reproducibility artifact associated with the SKB-Q manuscript. It is intentionally explicit about the boundary between checked-in executable code and the validated experimental evidence package.

## Status

**Partial reproducibility artifact.**

The `artifact/reproducibility-hf` branch contains an executable Stage-3 sensitivity-estimation procedure for OPT-125M, its configuration, and reproducibility documentation. It does **not** constitute a complete end-to-end implementation of every manuscript experiment.

## What a reviewer can reproduce from this branch

The supplied script can:

- load OPT-125M and WikiText-2 raw-v1;
- select calibration texts according to the script configuration;
- tokenize and evaluate a floating-point calibration loss;
- perturb eligible linear weights one node at a time;
- apply symmetric per-tensor, weight-only fake quantization;
- measure calibration-loss deltas for candidate bit widths; and
- save sensitivity rows and environment metadata as JSON.

## What is not established by this branch

The branch does not provide a verified end-to-end implementation that reproduces all reported Structure, Salience, Full, Random, or final mixed-precision allocation results in the manuscript. In particular, the presence of development/framework code elsewhere in the repository history or in other branches must not be interpreted as proof of complete replication of the final reported numbers.

The checked-in Stage-3 quantization semantics are also not evidence for the exact quantization semantics of every final manuscript experiment.

## Evidence boundary

The numerical results reported in the manuscript remain tied to the validated evidence package supplied with the submission. Existing code is not used to infer, regenerate, or replace a reported number unless that number has been independently validated against the corresponding evidence artifact.

## Recommended citation language

When referring to this repository in the manuscript or cover letter, use language such as:

> The project repository provides the available SKB-Q implementation components, experiment configuration, and reproducibility documentation. It should be interpreted as a partial reproducibility artifact; the validated experimental evidence package contains the machine-readable artifacts supporting the reported numerical results.

Avoid describing the repository as a complete replication package unless a future release adds and validates the missing end-to-end components.

## Release checklist

Before submission, verify that:

- [x] README states the artifact status accurately.
- [x] Repository structure exposes code, configuration, and documentation clearly.
- [x] Quantization semantics of the supplied script are documented explicitly.
- [x] The repository does not claim complete reproduction of manuscript experiments.
- [x] Mutable model revisions are flagged as a reproducibility limitation.
- [ ] A future complete release may add immutable model/tokenizer revisions and a pinned environment.
- [ ] A future complete release may add and validate the missing end-to-end allocation/evaluation implementation.
