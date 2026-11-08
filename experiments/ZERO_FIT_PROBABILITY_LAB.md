# Zero training variance does not imply low future risk

This synthetic experiment studies whether the origin lies in the convex hull of centered asset columns. For a fully invested nonnegative portfolio, this is exactly the existence of zero centered training variance. It says nothing about future investment performance. The separate constraint solver lab demonstrates that equally optimal training portfolios can have very different future variances.

With p columns in dimension d=T−1, general position and joint invariance under arbitrary individual column sign flips give membership probability
`1 − 2^(1−p) sum(comb(p−1,k), k=0,...,d−1)` (terms above p−1 vanish).
The primary reference is [Kabluchko, Vysotsky and Zaporozhets](https://arxiv.org/pdf/1510.04073), equation 6 and Remark 2.10. Independent Gaussian or independent Student-t coordinates satisfy the assumptions after centering. Common positive random row scaling preserves joint column sign-flip symmetry and conditional general position, despite dependence between columns. Correlated Gaussian columns generally do not satisfy this symmetry.

The fixed design is p=8, T=2,4,8,9 for Gaussian, independent t5 coordinates, and common-row-scale t5; plus p=2,T=2 Gaussian correlation −0.75,0,0.75. For the latter, centering leaves two Gaussian differences; membership is opposite signs. Their polar angle gives probability arccos(rho)/pi. This is a separate control law, not an extension of the independent-column formula. Each cell uses 2,000 independent draws, seed 733; no parameter search. The p8 analytic targets are 127/128, 99/128, 1/128 and zero. Helmert rows form an orthonormal basis of the centered observation subspace.

Each asset column is positively rescaled by its maximum absolute entry. Such rescaling preserves origin membership: feasible convex weights transform by inverse column scale and renormalization. It also preserves strict separation. We retain the original transformed matrix, normalized-space witnesses and residuals. Reported weights refer to normalized columns, not portfolio weights on the original data.

A feasibility LP is accepted only when returned weights are finite, nonnegative and sum to one, and normalized residuals are at most 1e−8. If not qualified, an independent LP maximizes strict separation margin with the separating vector bounded in infinity norm by one. Its actual minimum column inner product must exceed 1e−8. All other cases remain unresolved. These are floating-point numerical qualifications, not exact arithmetic certificates; extremely close boundary cases can remain unresolved or be numerically classified. No optimizer-success flag alone establishes membership.

Counts retain the full denominator. The membership bracket runs from accepted membership count/N to (membership+unresolved)/N; it represents numerical ambiguity, not a sampling confidence interval. No cases are discarded. Retained witnesses allow independent residual and separation audits. Analytic probabilities describe repeated sampling, not a guarantee that the observed frequency equals the target.

Run `OPENBLAS_NUM_THREADS=1 python experiments/zero_fit_probability_lab.py --repetitions 2000 --out results/zero-fit.json`. The root performance field is wall time only and excluded from scientific replay. Source hash and NumPy/SciPy versions are retained. No outside data, costs or trading actions are involved.

## Retained findings

The 30,000-draw run took 70.83 seconds and produced zero unresolved cases. Independent review recomputed every witness and independently classified 24,000 cases using direct sign, rank or nullspace geometry; all agreed. At p8,T4, Gaussian, independent t5 and common-row t5 observed membership frequencies were .7715, .7805 and .755 against the same analytic target .7734375. The largest descriptive standardized discrepancy across all 15 cells was 2.19; this is not an isolated significance claim or a simultaneous inference procedure.

Correlated p2,T2 controls observed .7735, .5015 and .232 at correlations −.75,0,.75, compared with analytic .76995,.5,.23005. Applying the independent-column .5 target to the correlated controls would be wrong. All p8,T9 observations were separated, consistent with the almost-sure full-rank result. Finite-sample agreement checks implementation; it does not prove the theorem.

The external compact summary adds pointwise Wilson interval envelopes over unresolved assignments, clearly separate from numerical ambiguity brackets. These are not simultaneous intervals over the 15 cells. Raw matrices and witnesses, source hashes, plotting source and full review evidence remain available in the private checkpoint.
