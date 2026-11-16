# Contribution boundary and closest controls

## Sources inspected

- Wang et al., *When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations for Dynamic 4-Bit Quantisation*, arXiv:2608.13365v1 (2026). HTML and TeX source inspected. The source archive contains paper/appendix TeX and figures, not the companion implementation described in appendix A.9. Targeted title/identifier searches did not locate that implementation. This is a statement about inspected material, not proof that no implementation exists. Source: https://arxiv.org/abs/2608.13365
- Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*, ICLR 2025, arXiv:2405.16406v4. Learned quantization-aware rotations and function-preserving transformer transformations precede the present experiments. Source: https://arxiv.org/abs/2405.16406 ; official code https://github.com/facebookresearch/SpinQuant
- Saxena et al., *ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals*, arXiv:2412.14363v2. PCA-based high-precision residual subspaces plus within-subspace rotations are prior mixed-precision/rotation mechanisms. Abstract inspected; this entry is not a full theorem audit. Source: https://arxiv.org/abs/2412.14363
- Kampeas et al., *Rotation Invariant Quantization for Model Compression*, arXiv:2303.03106. Rotation-invariant step selection is not the same as choosing a RoPE basis; title similarity alone is not algorithmic equivalence. Source: https://arxiv.org/abs/2303.03106 ; official code https://github.com/ehaleva/RIQ

## What each comparison establishes

| Control | Information and scope | Conclusion allowed |
|---|---|---|
| Identity + exact DP | Same weights, bits, byte budget and score | Effect of adding the finite basis search |
| Basis fixed from 4/4 + exact DP | Same grid, score and budget; basis cannot respond to chosen precision | Effect of conditioning a basis on precision, against this restricted control |
| Alternating basis/bit optimization | Same grid, score and budget; both variables updated | Whether the joint finite-grid solution adds anything over this standard stronger optimizer |
| Wang angle-rule transplantation | Calibration train text; same downstream weight RTN and byte DP | Controlled comparison of angle rules; not reproduction of dynamic W4A4KV4 or a victory over the entire published method |
| Full SpinQuant/ResQ pipelines | Different quantization scope; not run here | No end-to-end comparative superiority claim |

The SmolLM2 alternating control reaches exactly the same bit/basis map at 5, 6 and 7 Q/K weight bits. Pythia's score-table control does so as well. Consequently, neither the solver nor the measured gain over a frozen 4/4 basis demonstrates independent novelty. A new name, a standard DP proof, or additional repeated evaluation of identical tensors would not change that conclusion.

## Remaining scientific gate

The current evidence package can support a bounded empirical analysis of precision-conditioned basis selection. It cannot yet support the requested independent new-algorithm contribution or claim that the editor's novelty objection is resolved. A mechanism beyond standard joint optimization must be specified and tested against an equally informed optimizer; alternatively a substantial new theoretical or empirical insight must be established. This gate remains open and is not represented as a completed manuscript edit.
