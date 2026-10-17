# SKB-Q: Qwen + OPT frozen-forecast reproducibility work (author review)

**Status: research branch, not an archival release or a DOI.**

This branch records a reproducibility plan and reported validation targets for the SKB-Q manuscript. It does **not** establish an independent GPU rerun, validate hardware efficiency, or contain all raw trial artifacts.

- OPT-1.3B: 30 development configurations, 12 frozen heldout forecasts, C4 blocks 512–575, Spearman 0.972028, Ridge delta mean-NLL RMSE 0.0280345; budget 4.614307854 Linear-weight bits.
- Qwen2.5-1.5B: 30 random development configurations, 12 frozen heldout forecasts, C4 blocks 512–575, Spearman 0.860140, Ridge delta mean-NLL RMSE 0.00581142; budget 4.199719781 Linear-weight bits.
- OPT-to-Qwen direct zero-shot rank transfer failed (Spearman -0.755245). Models were separately trained on their **own** target-model configuration outcomes.
- Both runs evaluate fake-quantized 4/8-bit row-wise symmetric RTN. No packed-weight inference speed or memory gains were measured.

Closest novelty caveat: CASQ (Kričković & Kakani, *Electronics*, 2026, 15(19), 4573, https://doi.org/10.3390/electronics15194573) previously samples random configurations and fits an additive damage model for vision transformers; SKB-Q makes no 'first additive sensitivity estimator' claim.

**To reproduce:** download the author's reviewed evidence ZIP, run its `code/verify_holdouts.py` and `code/verify_manifest.py` for internal integrity, then execute the supplied Kaggle notebook workflows independently and compare full checkpoint outputs. Verify dataset/model revisions, licenses and all source data before public deposit. The 70-run v3 and 180-run circuit source archives must also be released separately for *full paper replication*.

The frozen forecast files and the new CASQ/Qwen-integrated manuscript are prepared locally for coauthor review. They are not uploaded to this public branch until author/license approval. The PIQA exploratory follow-up is not performed yet.

No Zenodo or Mendeley Data DOI was minted here. Replace a manuscript's 'pending' wording only after an actual immutable public release.

See `RELEASE_CHECKLIST.md` for the remaining gates.
