# Frozen real-model certificate selection (24 November 2026)

**Status: real pretrained-weight score and allocation, NOT a real-model PPL comparison.**

- Model: `HuggingFaceTB/SmolLM2-135M`, pinned revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Run: https://github.com/ahmadmohammadi80/skb-q-framework/actions/runs/45426233579
- Actual output artifact: https://github.com/ahmadmohammadi80/skb-q-framework/actions/runs/45426233579/artifacts/14936174024
- `certified_options.json` SHA-256: `ed8eb2bb69285c984d547fda3185a501ccdd2c38cc201027b355503731dd4aaa`
- `certified_allocation.json` SHA-256: `e2a7865a093de298d38069a83e545ee6d964d6dabbbe652bde9e8b0f0ddd58a8`
- Previous Gaussian stage2 score table (90 x 4 x 4), Git blob `6b87f39bbf66c74497f0f3b1ec334b6bef04ae16`, from `experiments/rope_20261116/stage2/scores.json`.

## Exact-budget choice difference

Each method optimized its own frozen criterion over **the same 90 GQA groups, four basis options, four Q/K bit options and 180 integer upgrade units**, implying identical Q/K patch format and budget **10,045,530 predicted packed bytes** including scales/maps. Only the previous benchmark physically exported Q/K patches; the latest GitHub Actions scorer writes allocation JSON, not new packed weights.

| Source of bit/basis decision | Q4/K4 | Q4/K8 | Q8/K4 | Q8/K8 |
|---|---:|---:|---:|---:|
| New norm-certified local attention bound | 12 | 44 | 0 | 34 |
| Earlier Gaussian-population attention score | 9 | 48 | 0 | 33 |

- Changes in bit-pair decisions: **46/90 GQA groups**.
- Changes in commuting rotation basis: **7/90 groups**.
- Certified bound evaluated on certified-optimal map: **8,498,424,114.96387** (sum of conditional upper bounds).
- **The same certified bound**, evaluated on the old Gaussian-selected map: **13,923,625,678.874586**.
- Reduction in the **nominal upper bound only**: **38.9640%**. This reduction follows from optimizing the new bound; it is **not** a 38.9640% decrease in measured real-model attention KL, NLL or perplexity. The bound may be extremely loose.

The two criteria are different (Gaussian average loss vs worst-case deterministic upper bound). Their raw objective values are not directly comparable. The comparison above evaluates both selected maps under exactly the new deterministic bound.

## What is still missing

1. Actual GPU evaluation of teacher and quantized models with identical WikiText blocks and physically equal Q/K packed bytes; the user has Colab and Kaggle notebooks. Existing test text was previously inspected: exploratory, not an untouched holdout.
2. Fair adapted-NSDS baseline on the same Q/K action space, clearly labeled **not the official published whole-model quantizer**; report both official-score rank orientations, do not pick only the weaker.
3. CASK-Q official end-to-end calibration run as a **separately disclosed unequal-information comparison** unless matching calibration access can be arranged.
4. A meaningful experimental connection from the certified upper bound to observed PPL (or truthful NO-GO). No independent-novelty or manuscript-ready claim until the comparison is complete.

**Conclusion:** a concrete, data-free, conditional-certificate algorithm now **selects a demonstrably different real pretrained-model bit map** from the earlier Gaussian method, with an exact objective-optimality proof and real weight-derived allocation artifact. This is an algorithmic step, not stage-four empirical success.
