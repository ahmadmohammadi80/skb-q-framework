# Proof scope and falsification targets — SKB-Q (2026-11-19)

Let X,Y be independent centered random vectors with independent coordinates, variances v_i>0 and fourth moments kappa_i*v_i^2. For a fixed two-token error E, expanding the identity S(E)=E[(X^T E X - X^T E Y)^2] gives

S(E) = tr(EV)^2 + tr(E V E V) + 2 tr(E V E^T V)
       + sum_i (kappa_i-3)*v_i^2*E_ii^2.

Proof: independence and E[Y]=0 cancel the cross term. The bilinear-square contributes tr(E V E^T V). The quadratic-form square has three Wick-style second-moment pairings, except that same-index fourth moments differ from 3*v_i^2 by (kappa_i-3)*v_i^2. This is a known fourth-moment identity, not novel.

Given a finite *predeclared* set of environments e and additive per-module losses L_(o,b,e)=S_e(E_(o,b)), exact discrete upgrade costs and a budget, vector DP retains per-cost nondominated partial score vectors. If a is dominated by b at equal cost, any identical completion r preserves a+r <= b+r, hence b cannot beat a in the minimax optimum. Induction over modules proves exactness. The number of nondominated vectors may be exponential.

The code tests 60 exact Rademacher enumerations and 250 independent exhaustive 5-module budget comparisons. Toy 3-module moment discrepancy: single-Gaussian selector worst across environments = 1041/16, robust selector = 129/4. Not a measured real RoPE attention loss, not a packed-byte comparison or perplexity result.

Major limitations: fixed E is **supplied**, not estimated by a real quantizer; no GQA/RoPE long-context score or actual model. The uncertainty family is not automatically valid for unseen hidden states; estimating it using calibration text voids a 'data-free' claim. Both moment expansion and Pareto DP are established prior math. Closest modern competitor threats are CASK-Q and NSDS; no formal novelty conclusion until fair experimental and theoretical contrasts.
