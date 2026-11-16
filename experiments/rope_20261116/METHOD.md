# SKB-Q precision-conditioned RoPE basis selection

This experiment implements a weight-only candidate for the query/key projections of SmolLM2-135M. It is a controlled experimental extension, not evidence that the editor's novelty objection is resolved. Earlier OPT archives are not inputs to this experiment.

## Decision rule

The checkpoint has 30 layers, nine query heads, three key heads, head dimension 64 and hidden dimension 576. Each of its 90 allocation groups contains one key head and the three query heads sharing it. All four projections in a group receive the same rotation, respecting grouped-query attention. Query precision is shared by the three query heads. The four bit choices are (4,4), (4,8), (8,4), and (8,8), ordered as query/key.

A basis consists of the same two-dimensional SO(2) rotation at each of the 32 split-half frequency pairs. The four angles are 0, pi/16, pi/8 and 3pi/16. This is a small restricted grid, not the full centralizer of RoPE. The rotation commutes with every relative RoPE rotation for this checkpoint.

For each group, basis and bit choice, symmetric per-output-row RTN is applied to the rotated weights. The score sums, over the three query heads, the mean squared Frobenius discrepancy between the original and quantized Q-transpose R K operators at relative offsets 0, 1, 16 and 64. No activation text is used to construct these scores. Input-distribution isotropy is implicit in the unweighted Frobenius surrogate; it is not established for the model's activations.

The identity control uses angle zero. The fixed-basis control chooses the angle minimizing the 4/4 score, then freezes that angle for all precision options. The conditioned rule chooses the minimum-score angle separately for every precision option. All three feed their option tables to the same exact-budget dynamic program. The 4/4 base cost is 74,752 bytes per group. The upgrades cost 0, 1, 3 or 4 units, each unit being 18,432 bytes. The budget permits exactly 180 units across 90 groups. The recurrence retains the least score at every reachable cost after every group; backtracking returns the bit and basis choices. This is standard multiple-choice knapsack dynamic programming, not a claimed new solver.

## What the mathematical guarantee says

An orthogonal basis U commuting with R satisfies U-transpose R U = R. Applying U to both Q and K therefore preserves the full-precision bilinear operator. Sharing U across the key head and its associated query heads preserves every attention score, and hence the full-precision network function in exact arithmetic.

For a finite basis grid, minimizing over bases independently for every bit option before the budget dynamic program gives the global minimum of the specified separable surrogate over that grid and feasible allocations. Proof: at fixed bits, each group's basis affects only that group's score and does not change storage; thus its minimum can be taken locally. Induction over groups then establishes the usual dynamic-programming optimum. Identity and the fixed control are feasible members, so the conditioned surrogate cannot be greater. Neither this result nor full-precision invariance implies improved perplexity, novelty, or superiority over all fixed-basis algorithms. In particular, for one budget the conditioned solution itself defines a fixed basis assignment that an adequately optimized joint competitor could reproduce.

## Storage and implementation boundary

Quantized values have 4 or 8 bits, and each row has a float32 scale; symmetric quantization needs no stored zero-point. The experimental format budgets one byte per group for its precision code, including padding. Basis rotations are folded into the weight values and require no inference-time basis matrix. The matched total is 10,045,530 bytes for Q/K, including scales and the 90-byte precision map. This representation was exported for the actual model weights. Each Q/K patch file has exactly 10,045,530 bytes; all 180 projection groups per method round-tripped exactly to the dequantized model values. File hashes are in packed_qk_manifest.json. Inference itself uses dequantized float32 weights. Other parameters, activations and caches remain float32. These results do not measure packed inference latency or process memory, and do not describe a six-bit whole model.

## Published rotation comparator

Wang, Luo, Xu and Cheung, When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations for Dynamic 4-Bit Quantisation (2026), arXiv:2608.13365, Eq.3 and Appendix A.2.1, supplies the closest inspected angle rule. We pool Q/K first and second moments across heads on 16 training blocks, preserving the physical 3:1 Q:K observation ratio. We construct the finite-position averaged covariance at length 512 and apply the paper's analytic variance-equalizing angle per frequency pair, shared across heads in each layer. The angle is folded into Q/K weights. Its four precision scores and storage allocation are evaluated using the same RTN and dynamic program as our controls.

This is a direct controlled comparison of rotation rules, with a transplanted published estimator. It is NOT a reproduction of the paper's complete dynamic W4A4KV4 pipeline, its Hadamard composition, or its published checkpoints. The comparator sees training activations; our grid selector uses weights only. The scope difference must be retained in tables and claims.

SpinQuant (ICLR 2025, arXiv:2405.16406) already studies learned function-preserving rotations. GaugeQuant (arXiv:2607.20757) learns bases during training. TORQUE (arXiv:2609.36032) jointly chooses high-precision retention around rotations. Consequently, neither rotation, basis learning, nor the generic combination of rotation and precision allocation can be asserted as new. The possible remaining distinction is the precise finite-grid, GQA-coupled, weight-only operator objective with bit-conditioned choices; independence and significance of that combination remain to be established.

## Evaluation protocol

Model and tokenizer: HuggingFaceTB/SmolLM2-135M, revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2. Dataset: Salesforce/wikitext, wikitext-2-raw-v1. Join all test text entries with two newline characters, tokenize once, then evaluate all 595 complete non-overlapping 512-token blocks. Each block predicts 511 tokens, giving 304,045 scored tokens; 338 final tokens are discarded. Context resets at each block. Perplexity is exp(mean block NLL); block lengths are equal. Evaluation runs in eval/inference mode on CPU with float32 computation.

The first 16 test blocks were inspected as a pilot before expansion; no angles, budgets or decision rules were changed afterwards. The full-test evaluation includes that pilot prefix and is not an independent second dataset. Published-rule calibration uses only the first 16 training blocks and never the test set. Bootstrap intervals describe paired corpus-block variation, not independent model retraining or multiple calibration seeds. Raw losses, allocations, angles, environment versions and text hashes are retained.

## Reproduction

Install CPU torch with `python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu`, then run `python -m pip install -r requirements.txt`. The executed environment is recorded in environment.json. Run run.py to generate the score table and pilot; run run_full.py for the full four-control evaluation; run compare_published.py for the transplanted published rule; run validate.py, check_model.py and export_qk.py for checks and real packed Q/K patch export; finally run analyze.py to validate complete records and recompute summary.json. The scripts require network access on first use to obtain the pinned model and dataset. Cached execution can use HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1.
