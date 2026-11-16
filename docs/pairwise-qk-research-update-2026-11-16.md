# SKB-Q pairwise Q/K research update

The proposed extension quantizes Q and K projection weights. For column-vector inputs and relative RoPE displacement r, the reference attention-logit operator is `W_Q.T @ R_r @ W_K`. Evaluate the difference between the exact quantized and reference operators in each of the four bit configurations. Define the interaction score as `D_84 + D_48 - D_44 - D_88`, where D is the squared operator discrepancy under the declared norm and displacement weighting.

## Verified analytical example

With three equal-cost groups Q, K and V, individual upgrade scores 3, 2 and 4, and a budget for two upgrades, the additive objective selects Q and V (score 7). A Q/K interaction bonus of 3 selects Q and K (score 8) at the same bit cost. The interaction in this example is stipulated. The example establishes an allocation difference under the surrogate objective.

## Validation status

The pairwise variant has not been benchmarked on a held-out architecture. No new perplexity or latency result is claimed. Compare additive scoring, first-order pairwise scoring and exact-operator pairwise scoring with the same solver, matched storage including quantization metadata and identical evaluation tokens.
