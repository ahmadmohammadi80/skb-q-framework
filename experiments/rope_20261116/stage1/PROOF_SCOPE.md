# What the algorithm actually guarantees

For group g, precision choice j, and finite basis a, let D[g,a,j] be the explicitly computed weight-operator Frobenius surrogate. A choice costs c[g,j] bytes, independent of basis because rotations are folded into weights. Then

min_{j_g,a_g: sum c[g,j_g] <= B} sum D[g,a_g,j_g]
= min_{j_g: sum c[g,j_g] <= B} sum min_a D[g,a,j_g].

Proof: every feasible joint assignment lower-bounds the right side. Selecting a minimizer separately for each selected precision attains that lower bound without changing costs. A multiple-choice knapsack DP solves the remaining finite problem. The present experiments enforce exact use of the declared budget; <= B is obtained by taking the best terminal cost. This elimination and DP are standard optimization, not a new theorem or an independent novelty claim. The guarantee is for the recorded finite-grid surrogate, not perplexity or continuous rotations.

For full or partial split-half RoPE, the implemented U rotates each rotary pair by the same angle and is identity outside the rotary subspace. Each pair rotation commutes with the corresponding RoPE pair; hence U^T R_r U=R_r. Rotating both Q/K biases is necessary when biases exist. Thus the unquantized attention logits are unchanged in exact arithmetic. Numerical model-level checks test the implementation.

The score uses weight matrices only. In Pythia, the rotated biases preserve the unquantized mapping but are not included in the score's error objective. Neither isotropic input statistics nor the four sampled offsets are claimed to describe all real activations/context lengths. This limits the proxy's predictive power.

Alternating basis minimization and exact bit allocation is a stronger control than freezing the basis optimized at 4/4. Identical final maps establish identical quantized tensors; rerunning inference on those identical tensors does not constitute an independent new experiment.
