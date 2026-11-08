# Reading the offline inference experiments

These labs isolate different ways a research result can look stronger than its
evidence. They are synthetic controls and reusable research examples, not an
OpenQuant promotion policy. None establishes that a trading strategy is profitable.

## Start with the target and the selection procedure

Expected excess return, a specified Sharpe threshold, the probability of a
positive nonzero return, and expected bounded score are different targets. A
sign procedure can be well calibrated while losing money when negative returns
are larger. The sign labs intentionally make no claim about return magnitudes.

Write down the family of candidate functions, preprocessing, selection rule,
observation units, and reporting time before running an inferential comparison.
The following map identifies experiments relevant to those choices; it does not
select a procedure using the best observed result.

| Research question | Experiment | What the control establishes or exposes |
| --- | --- | --- |
| Does selecting the best candidate inflate its observed score? | [One-shot validation](VALIDATION_LAB.md) | Training-only selection versus contaminated selection, under explicit synthetic families. |
| Does feedback change the candidate family? | [Adaptive holdout](ADAPTIVE_HOLDOUT_LAB.md) | Reusing validation feedback to construct new blends; a final independent audit remains separate. |
| Can preprocessing leak without future labels? | [PCA preprocessing](PREPROCESSING_LEAKAGE_LAB.md) | Fitting a transform to future feature rows can help or hurt. Full-component ridge supplies an invariance control. |
| Is a selected sign claim calibrated? | [Sign audit](ADAPTIVE_SIGN_AUDIT.md) | Exact binomial reference for independent sign trials; serial dependence breaks its premise. |
| Can a fixed family share a mean bound? | [Bounded finite family](BOUNDED_FINITE_FAMILY_LAB.md) | Empirical Rademacher and Hoeffding controls for known bounded scores and IID rows. |
| Does grouping observations create independence? | [Cluster sign](CLUSTER_SIGN_LAB.md) | Independent blocks versus a continuous path and shared factors. Arbitrary grouping is insufficient. |
| What interval covers a selected candidate's sign probability? | [Selection intervals](SELECTION_INTERVAL_LAB.md) | Fixed-family simultaneous correction, independent audit, hidden-family and repeated-row failures. |
| Does repeatedly checking a result change its error rate? | [Optional stopping](SEQUENTIAL_SIGN_LAB.md) | Fixed-time tests versus predetermined betting rules; conditional fairness differs from marginal fairness. |
| What if both candidates and observation times are searched? | [Multiple sequential streams](MULTIPLE_SEQUENTIAL_LAB.md) | Global discovery and candidate-specific attribution require different claims; cross-stream dependence is explicit. |
| How much does estimated covariance distort apparent risk? | [Covariance estimation](COVARIANCE_ESTIMATION_LAB.md) | Unrestricted Gaussian sample weights have exact reference laws, including infinite risk moments; known truth scores the fit but does not train it. |
| Can positive sign evidence coexist with negative expected payoff? | [Payoff targets](PAYOFF_TARGET_LAB.md) | Identical sign paths, different win/loss magnitudes and costs; bounded stopping preserves the distinction. |
| What information does a separate audit cost? | [Equal observation budget](SELECTION_BUDGET_LAB.md) | Selection accuracy, interval width and claims at equal row budgets; candidate-evaluation costs differ. |

## Further robustness and data-availability controls

| Research question | Experiment | Main qualification |
| --- | --- | --- |
| Can mean inference withstand heavy tails? | [Finite-variance mean bounds](MEAN_ROBUSTNESS_LAB.md) | Variance bounds are assumptions; conservative widths can conceal violations. |
| What if the sample never sees a rare outcome? | [Rare-event uncertainty](RARE_EVENT_UNCERTAINTY_LAB.md) | Empirical uncertainty can collapse; known-support controls use additional information. |
| How much remains unidentified after missing observations? | [Partial identification](PARTIAL_IDENTIFICATION_LAB.md) | Sharp population bounds and finite-sample outer confidence sets are different; structural width need not shrink with sample size. |
| Can weighting recover a filtered population? | [Informative missingness](INFORMATIVE_MISSINGNESS_LAB.md) | Known observation probabilities and positivity are substantive assumptions; observational equivalence limits identification. |
| Is an optimizer difference substantive or a tie-breaking artifact? | [Constraint solvers](CONSTRAINT_SOLVER_LAB.md) | Exact feasible-solution reuse is distinct from choosing among nonunique training optima. |
| How surprising is zero fitted variance? | [Zero-fit probability](ZERO_FIT_PROBABILITY_LAB.md) | Geometric symmetry assumptions differ from covariance-moment assumptions; numerical witnesses are not certified arithmetic. |
| Do constraints help outside the Gaussian reference model? | [Covariance robustness](COVARIANCE_ROBUSTNESS_LAB.md) | Future distribution shifts, estimator failures and numerical equivalence remain explicit. |
| Were training labels available when the model was fitted? | [Delayed labels](DELAYED_LABEL_LAB.md) | Provider release violations and actual future-content overlap are audited separately. |
| Which revision was actually available at a decision? | [Point-in-time lookup](POINT_IN_TIME_LAB.md) | Event time alone is insufficient; availability, missingness and historical provenance are separate contracts. |

## Assumptions do not transfer automatically

**Across candidates and across observations are separate issues.** The
Bonferroni argument for a fixed family does not require independent candidates.
The component Clopper–Pearson intervals still require the stated IID observation
model. The budget lab's finite-sum calculations add independence between candidate
counts; its efficiency conclusions therefore have a narrower scope.

**An independent audit requires both separation and a suitable observation model.**
Generating discovery and audit samples independently does not repair repeated
rows inside either sample. Adjacent historical periods are not independent just
because their timestamps do not overlap. The labs manufacture independence where
they rely on it; they do not diagnose it in market data.

**The number of logged trials is not necessarily a fixed family size.** A family
of functions specified before evaluation differs from functions constructed in
response to evaluation feedback. Counting the realized adaptive queries does not
by itself justify the fixed-family correction. Duplicate observed columns are not
a reason to reduce the correction: observed agreement need not mean functions
were identical before seeing data.

**Coverage and usefulness differ.** A vacuous [0,1] interval may be valid but
uninformative. A method can have narrower intervals and select weaker candidates.
Small effective sample sizes, unattainable discrete rejection thresholds, and
numerical fail-closed intervals are retained as results. They are not silently
replaced with a more favorable statistic.

**Separate guarantees do not combine for free.** Looking at multiple procedures
and reporting only the favorable one is another selection step. Pointwise Monte
Carlo uncertainty across many settings is not a simultaneous uncertainty band.
The studies report their prespecified grids and counterexamples, including
settings that contradict a preferred narrative.

## Reproduce before extending

Each report records configuration, code hashes and numerical-library versions.
Use the [fresh-process replay command](README.md#replay-saved-offline-evidence)
with its matching source snapshot. Exact replay checks reproducibility, not
provenance authenticity or applicability of model assumptions. A mismatch on a
different numerical platform is not alone evidence of a bug or misconduct.

Keep extensions distinct from the frozen evidence: state a new question, preserve
a simple baseline and a deliberately invalid control, declare the grid before
interpreting it, then retain every result and an independent calculation where
possible. Runtime and candidate-evaluation counts are useful costs; they are not
substitutes for acquisition costs, transaction costs or economic utility.
