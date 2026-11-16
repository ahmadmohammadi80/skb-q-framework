# Exact Gaussian expectation of centered causal attention error

This is a proposed scoring extension, not an established independent novelty claim. Centered causal logit error, the sharp 1/4 attention-KL bound and exact prefix-moment evaluation are prior CASK-Q results. Our implemented extension analytically integrates that error under an explicitly restricted Gaussian input model and evaluates the full RoPE context by a reusable spectral kernel.

## Domain and decision rule

A head has even dimension d and bias-free Q,K in R^(d x p). Token inputs x_1,...,x_L are independent N(0,I_p) **for this score model only**. Real LLM hidden states need not satisfy this assumption. Teacher q_i=R_i Q x_i, k_j=R_j K x_j. Student uses quantized A,B. Consequently the error operator for lag r=i-j is E_r=A^T R_(-r) B-Q^T R_(-r) K. Negative causal offsets matter; an ablation separately corrects the earlier positive-offset proxy.

For each group, each of the original four commuting bases and each of four Q/K precisions, compute the score below and pass the resulting option table to the unchanged exact-budget DP. Thus the possible contribution is the score evaluator, not a new DP or a new rotation family. GQA groups sum the scores of their three associated query heads.

## Expected centered error

Let delta_i contain the i causal logit errors, including 1/sqrt(d), and H_i=I-11^T/i. Define J=(1/(4L)) sum_i E||H_i delta_i||^2. For r>0, E[(x_i^T E_r x_(i-r))^2]=||E_r||_F^2. For the self key,

s_0 = E[(x^T E_0 x)^2] = ||E_0||_F^2 + tr(E_0^2) + tr(E_0)^2.

This follows from Gaussian fourth moments, and remains valid when E_0 is not symmetric. Off-diagonal cross moments of distinct causal logit errors vanish: condition on the shared query and integrate the independent zero-mean keys; cross moments involving the self key also vanish. Therefore

J = [w_0 s_0 + sum_(r=1)^(L-1) w_r ||E_r||_F^2]/(4d),
w_r = (1/L) sum_(i=r+1)^L (1-1/i).

At L=1 every centered logit vector is zero and J=0, as tested. A Frobenius-only self term is generally incorrect: for E_0=diag(1,-1), its value is 2 but the Gaussian self second moment is 4.

The prior global bound KL(softmax(z)||softmax(z+delta)) <= ||H delta||^2/4 follows from log-sum-exp smoothness, since its Hessian has operator norm at most 1/2 and additive constants are softmax-null. Consequently the average **expected attention-row KL under this Gaussian model** is at most J. This is not a bound on model perplexity, output error, or real-data attention KL without the Gaussian assumptions. The global 1/4 bound is credited to existing work, not claimed here.

## Spectral evaluation without per-candidate context traversal

Diagonalize R_r=V diag(exp(i lambda r)) V*, with lambda=(omega,-omega). Define the reusable kernel

G_ab = sum_(r=1)^(L-1) w_r exp(-i (lambda_a-lambda_b) r).

Then sum_r w_r R_(-r) M R_(-r)^T = V [G elementwise (V* M V)] V*. Each squared Frobenius term uses only d x d Gram matrices:

||A^T R B-Q^T R K||_F^2
= tr(AA^T R BB^T R^T) + tr(QQ^T R KK^T R^T)
  - 2 tr(QA^T R BK^T R^T).

Thus full-context averaging uses three kernel contractions. The self term also uses d x d matrices:
tr(E_0)=tr(BA^T)-tr(KQ^T),
tr(E_0^2)=tr((BA^T)^2)+tr((KQ^T)^2)-2 tr(BQ^T KA^T).

The kernel is positive semidefinite: it is a weighted sum of outer products of Fourier vectors with nonnegative weights. Its construction costs O(L d^2), once per context/frequency configuration. Each candidate then costs O(d^2 p+d^3), independent of L, instead of O(d^2 p+L d^3) for explicit Gram-matrix traversal. The implementation uses dense basis changes; exploiting the pairwise sparsity could further reduce constants but is not claimed as implemented.

## Relationship to CASK-Q

CASK-Q's official exact_prefix_error returns the empirical centered squared logit error divided by the number of allowed pairs, L(L+1)/2. Therefore its Gaussian expectation equals 8J/(L+1). The official-core comparison verifies this normalization and Monte Carlo agreement. This is an exact population evaluator for a restricted input distribution, not a competing empirical calibration objective or a full reproduction of CASK-Q training. Its assumption dependence and absence of real calibration data are material differences, not proof of superior quality or independent novelty.
