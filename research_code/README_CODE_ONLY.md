# SKB-Q: research code (code-only distribution)

This folder distributes Python analysis and evaluation code connected to the SKB-Q research project. **It does not distribute** training/evaluation measurements, allocation manifests, model weights, experimental archives, private documents, unpublished manuscripts, or any user conversation content.

## Contents

- `run_additivity_audit.py`: retrospective additive-model comparison and cost-ledger checks. Read input ZIPs from `SKBQ_DATA_ROOT` (defaults to the current working directory).
- `verify_basc_validation.py`: offline validation of a previously executed BASC-20 experiment using an externally supplied frozen manifest and GPU result ZIP.
- `novelty_gate.py`: exploratory 20/10 development-map resampling. Needs `SKBQ_Qwen_Stage1_Development.zip` in the script folder or current directory.
- `sparse_interactions_screen.py`: exploratory adjacency-interaction screen on local development and holdout archives. Set `SKBQ_DATA_ROOT` or put original archives beside the script.
- `BASC20_Validation_CODE_ONLY.ipynb`: Kaggle GPU evaluation notebook that **requires** the original frozen manifest supplied privately via `SKBQ_BASC_MANIFEST`.
- `Prospective_Ridge_Selection_CODE_ONLY.ipynb`: original-style prospective Qwen Ridge evaluation notebook, but with its frozen allocation manifest **externalized** via `SKBQ_DECISION_MANIFEST`.

## Installation

Use a project-specific Python environment. For offline scripts install `numpy`, `scipy`, and `scikit-learn`; the Kaggle notebooks additionally use `torch`, `transformers`, `datasets`, and `huggingface-hub`. Notebook package versions and model/dataset revisions should be read from the notebook cells and frozen protocol, and should not be silently upgraded when replicating.

## Limitations and scientific status

1. Original frozen map manifests and input data are intentionally **not included** in this code-only publication. Code release alone is **not** sufficient for independent end-to-end reproducibility.
2. The exploratory additivity and novelty scripts reuse previously observed data. Their outputs are not pre-registered independent confirmation.
3. The evaluation uses simulated row-wise RTN; no packed-kernel speedup, energy benefit, physical-memory savings, GPTQ or AWQ transfer is demonstrated by these scripts.
4. Code syntax was checked; **independent GPU reruns are not asserted** by this publication.
5. No license is granted merely by the presence of public source. The copyright holder must select an explicit reuse license separately if desired.

## Locally provided files

Do not commit private ZIPs, JSON manifests, model tokens, credentials, personal correspondence, submitted drafts, or generated outputs. Prefer private/local mountpoints and keep those paths out of version control.

## Research ethics

Report positive and negative results, record comparisons at matched effective bit budgets, and distinguish post hoc exploration from frozen prospective tests. No manuscript is deposited or submitted by this code publication.
