# SKB-Q numerical evidence — public dataset (code/data split)

This directory contains **machine-readable numeric observations** released with the author's permission, as supporting research data for an experimental study of 4/8-bit attention-group allocation. It does **not** include any manuscript, private conversations, passwords/tokens, source corpus text, GPU console logs, or model weights. It was generated from the named source archives; original ZIP hashes appear in `DATASET_MANIFEST.json`.

## Contents

- `opt_v3_70.jsonl`: **70** controlled OPT protocol-v3 configurations, 35 each for OPT-125M and OPT-1.3B. Each row includes 243 per-block NLL observations, a linear-parameter-weighted average bit width, and the originally logged T4 runtime. Uniform-4/8 controls are **not** budget matched to the 4.61-bit policies.
- `opt_ridge12.jsonl`: **14** additional OPT-1.3B confirmatory checkpoint policies; a **different experiment** from OPT v3.
- `qwen_c4_development.jsonl`: **31** policies (including controls) on the C4 development protocol, used to train a surrogate from 30 random maps.
- `qwen_c4_holdout.jsonl`: **13** held-out C4 policies (including controls); these are *prediction* tests, not an unrestricted top-K allocation optimization.
- `qwen_wikitext_test_prospective.jsonl`: **7** prospectively selected/comparator policies on WikiText-2 test.
- `qwen_wikitext_test_baselines.jsonl`: **7** Stage B controls on WikiText-2 test, including FP16, Uniform4 and Uniform8 (different budgets).
- `qwen_wikitext_validation_basc.jsonl`: **9** BASC-20 and comparator policies on WikiText-2 validation. Do not label this another WikiText test set.
- The offline audit package additionally contains `policy_summary.csv`, a one-row-per-policy index. It is **not** part of this repository release and is not required to recompute the individual policies.
- `DATASET_MANIFEST.json`: parent archive SHA-256 digests and scope limitations.

These published numeric JSONL files permit recomputation of the reported per-policy NLL and perplexity and paired blockwise descriptive contrasts. Hashes and allocations are observational records and do not constitute third-party timestamped preregistration. Paired block bootstrap intervals are descriptive and are not independent-document inferential guarantees.

## Explicit methodological limits

**Do not** treat per-block next-token losses as independent documents or samples, treat weight-only surrogate scoring as cost-free, or infer real packed quantization speedups from fake quantization. The source model and datasets must be obtained independently under their own licenses and usage policies. Source archive SHA-256 hashes let a holder of original private ZIPs verify correspondence; they do not mean that these ZIPs are published.

## Code

Code-only reproducibility scripts and notebooks are in `research_code/` on the same release branch. Some notebooks require frozen allocation maps and protocols; the present JSONL release provides selected maps and numeric outcomes but is **not** a promise of fully independently runnable end-to-end GPU replication without further environment setup.

## Citation / license

Research data attributed to SKB-Q authors. Dataset DOI: **not assigned**. Before citing in a final manuscript, use the permanent version or repository commit identifier. No public license has been selected yet; public GitHub access by itself does not grant permission for reuse beyond applicable law.

## Publication integrity

The seven numerical JSONL files are byte-for-byte identical to their reviewed local source versions. `SHA256_DATA_FILES.txt` records their SHA-256 digests and the data manifest digest. This repository deliberately excludes the auxiliary CSV index because a connector text export does not preserve its exact original numerical formatting. No archived source ZIP is implicitly published.
