# Certified-RoPE-Budget: theorem, algorithm and falsification scope

**Research candidate, 2026-11-24.** This is a new *conditional local-attention* objective, not a claim of globally established independent novelty, model-level safety, demonstrated PPL improvement, or a completed end-to-end CASK-Q comparison. It does not alter any earlier prototype.

## Specific problem

Existing SKB-Q full-context Gaussian-population scorer relies on iid Gaussian hidden states, which are not guaranteed in real LLMs. Solve the following narrower problem without calibration data:

**Among a fixed list of GQA-consistent Q/K 4/8-bit and commuting RoPE basis choices, minimize the sum of *distribution-free, finite-context conditional per-head attention-row KL upper bounds* under an exact physically stored Q/K byte budget.**

The bound is a *local same-input* attention comparison: teacher and quantized Q/K receive the SAME normalized token vectors. It is NOT an end-to-end bound where quantization of previous layers changes the hidden vectors.

## Theorem: samplewise deterministic attention KL bound

For a causal head with input width p, even head dimension d, length L, teacher Q,K and quantized A,C all in R^(d x p), let normalized token vectors x_1,...,x_L obey ||x_i|| <= B. The RoPE relative-offset matrix is R_{j-i}. Define for nonnegative causal lag r:

E_r = A^T R_{-r} C - Q^T R_{-r} K.

Use standard attention logits 1/sqrt(d), causal masks, same positional embeddings and token inputs. Let KL_i denote teacher-to-student softmax attention KL for row i, supported on keys j<=i. Then **for every token sequence**, irrespective of distribution, dependence, covariance or non-Gaussianity,

(1/L) sum_{i=1}^L KL_i <= [B^4/(4d)] sum_{r=0}^{L-1} [(L-r)/L] ||E_r||_F^2.  (1)

For a standard RMSNorm scale gamma with epsilon>=0 and output width p, we may set B = sqrt(p) ||gamma||_infinity *without observing activation data*.

**Proof.** For j<=i the student-minus-teacher logit is delta_ij = x_i^T E_(i-j) x_j/sqrt(d). Cauchy-Schwarz and ||E||_2<=||E||_F give |delta_ij| <= B^2 ||E_(i-j)||_F/sqrt(d). For any z,delta, KL(softmax(z)||softmax(z+delta)) is the logsumexp Bregman divergence. The logsumexp Hessian diag(p)-pp^T has spectral norm at most 1/2, hence this KL is <=||delta||_2^2/4 (the latter inequality is known from work including CASK-Q; it is NOT claimed here as new). Summing the squared-logit bound for row i, averaging over i, and noting that lag r occurs in exactly L-r causal pairs proves (1). For RMSNorm, ||gamma * v / sqrt(||v||^2/p+epsilon)|| <= sqrt(p) ||gamma||_infinity. QED.

**Important:** neither (1) nor the subsequent knapsack optimum bounds final model perplexity, student-vs-teacher hidden-state drift, or absolute downstream losses.

## Spectral RoPE algorithm

For nonnegative weights w_r=(L-r)/L, evaluate F = sum_r w_r ||E_r||_F^2 via three Gram contractions:

||E_r||_F^2 =
tr(AA^T R CC^T R^T) + tr(QQ^T R KK^T R^T)
-2 tr(QA^T R CK^T R^T).

RoPE has eigenvectors V and Fourier phases lambda; precompute
G_ab = sum_r w_r exp(-i(lambda_a-lambda_b) r).

Then each weighted trace is evaluated in the V basis using an elementwise multiplication by G. This is a standard Fourier contraction, not itself a new spectral theorem. The implementation 'certified_score.py' uses O(L d^2) one-time kernel construction and O(d^2 p + d^3) arithmetic per option after Gram formation, instead of explicit L offset traversals. Its independent 'explicit' reference validates the formula.

For each of 90 GQA groups in pinned SmolLM2-135M, choose one of four precision pairs (Q4/K4, Q4/K8, Q8/K4, Q8/K8) and one of four commuting rotation angles. Sum three head certificates in each group, using the *layer-specific frozen input RMSNorm gamma*. A standard multiple-choice exact-cost dynamic program minimizes the sum of scores over the finite grid. This conditional finite decision optimum is provable by induction on groups; it is NOT a novel general optimizer. Commuting basis U applied to both Q and K preserves the full-precision RoPE bilinear operator (UQ)^T R (UK)=Q^T R K, and within GQA the same U is shared among associated query and key heads.

Physical Q/K budget in the provided execution kit:
90 * 74,752 + 180 * 18,432 + 90 one-byte maps = **10,045,530 bytes**, including float32 per-row scales. Packed patch files are real encoded Q/K data, while inference uses dequantized FP32. No speedup is claimed.

## Test and research gates

- Independent spectral-vs-explicit RoPE contraction; analytic RMSNorm norm inequality; test of full-precision commuting-basis invariance; *samplewise softmax KL inequality on correlated non-Gaussian input vectors*; tiny model generation; physical 90-group byte packing and exact-budget DP.
- Full matched data/model/tokenizer/budget GPU evaluation is required against NSDS *both ranking directions*, old Gaussian self/no-self, identity, random and FP32 anchor.
- The official NSDS scorer requires documented mechanical patches and is mapped to a Q/K-only equal-budget action space; this is **not** reproduction of its published full-model bit-allocation method.
- CASK-Q uses calibration-dependent activation information. Its prior centered-error and 1/4 KL objective are acknowledged; a model-level CASK-Q comparison must declare or match its extra information cost.
- Model WikiText test data were examined earlier; any result there is exploratory until independently confirmed.
- Code and CPU theorem/unit tests are complete in the downloadable Colab/Kaggle execution kit. **No GPU PPL result is fabricated here.**

This guarantees a mathematically specified conditional optimization problem and implements an algorithm solving that problem. Its relation to wider prior art and its empirical advantage remain open research questions.
