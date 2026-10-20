# Final locked Qwen risk-aware allocation experiment (2026-10-20)

This public folder contains the **pre-outcome** frozen policy maps, scientific decision rules, a runnable Kaggle/Colab GPU notebook, and an offline results verifier.

**Freeze content SHA-256 (canonical JSON)**: `188866e4c113a60cbee231473984aa52b2b8bcc1399269cb803a481f532a63a6`

Seven policies, all with 14 upgraded attention groups of 28 under the same 4/8-bit simulated RTN budget, will be evaluated on previously unmeasured WikiText-2 test blocks **64-127**. Earlier measured test blocks were **0-63**. The risk method was designed with knowledge of earlier studies; do not claim it was invented independently of their outcomes. The target is a new token portion, **not documented to be document-independent**. Zero target GPU results are part of this frozen publication.

1. Upload `SKBQ_Qwen_FINAL_FROZEN_2026-10-20.json` into Kaggle/Colab alongside the notebook or point `SKBQ_FREEZE_MANIFEST` at it.
2. Enable GPU; execute the notebook from the top. It enforces the manifest hash, model revision, exact bit budget, and target token lock, and checkpoints each policy.
3. Share the complete `SKBQ_Qwen_Final_Locked_Trial.zip` output, even if the new method loses; run `python verify_frozen_trial.py SKBQ_Qwen_Final_Locked_Trial.zip` to check internal arithmetic.

**Primary endpoint:** mean target NLL of `RiskCVaR_30` strictly below both `LassoCV_30` and `OLS_30_CASQ_family`. The latter is only a CASQ-family unregularized additive proxy, **not full CASQ**. Secondary descriptive endpoint: empirical CVaR_0.75 of blockwise policy-minus-Late NLL. Do not retune, move the prefix or rerun for a desired outcome. A positive result on this one setting is not proof of general or full-CASQ superiority, real packed-kernel gain, or journal acceptance.

Source research code and released numeric training observations are also on the parent repository. Full historical source archives, third-party corpus/model files, and private correspondence are not published here.
