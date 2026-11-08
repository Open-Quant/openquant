# Bounded finite-family mean-bound experiment

Status: private experimental research scaffold. Owner: OpenQuant maintainers.
Audience: researchers checking assumptions before adding a selection-adjusted metric.

## What was verified, and what was not

Paleologo's [author page](https://linktr.ee/paleologo) links his public
[September 16, 2024 BAML backtesting slides](https://docs.google.com/presentation/d/1AoS83uXEqf9XRraO4ZwvNKPKC9b0JICnWn5MCR83H-0/htmlpresent).
Those slides discuss Rademacher complexity and simultaneous performance bounds.
The equations are embedded images; their bytes could not be retrieved in this
environment. The exact RAS constants, dependence assumptions and proofs remain
unverified, as does the book's full text. No unofficial book copy was used.

**This implementation is not Paleologo RAS.** It uses the bounded-IID theorem in
Mehryar Mohri's [official lecture slides, pages 4–9](https://cs.nyu.edu/~mohri/mls/ml_learning_with_infinite_hypothesis_sets.pdf),
which give the definition, generalization inequality and proof. The
[author's textbook page](https://cs.nyu.edu/~mohri/mlbook/) also identifies legitimate
online editions, but no inaccessible text is treated as verified evidence.

## Verified equation and explicit translation

Let a fixed, finite family `f_1,...,f_K` take values in `[0,1]`. Draw `n` observation
rows independently from the same distribution. Write `X_ij = f_j(Z_i)` and
`mu_j = E f_j(Z)`. Columns may depend arbitrarily on one another within a row.

Use the **nonabsolute, no-factor-two** empirical complexity convention:

```text
Rhat(X) = E_eps [ max_j (1/n) sum_i eps_i X_ij ],
eps_i independent and uniform on {-1,+1}.
```

Mohri's empirical theorem bounds population means above by sample means plus
`2 Rhat + 3 sqrt(log(2/delta)/(2n))`. Our lower form follows by applying it to
`1-f_j`: sign symmetry and the zero-mean common offset leave Rhat unchanged.
Thus, simultaneously for every candidate, with probability at least `1-delta`:

```text
mu_j >= mean(X[:,j]) - 2 Rhat(X) - 3 sqrt(log(2/delta)/(2n)).
```

The original [Bartlett–Mendelson paper (2002), Definition 2](https://www.jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf)
uses an absolute supremum and factor two in its complexity definition. Its constants
must not be substituted into the different convention used here.

## Monte Carlo correction: an additional derivation

For an independently drawn sign vector, let `a_j = mean_i(eps_i X_ij)` and compute
`Z = (max_j a_j - min_j a_j)/2`. This pairs a sign vector with its negative.
By symmetry, `E Z = Rhat`. Also, `0 <= Z <= 1/2`: every difference `a_j-a_k`
has absolute value at most `mean_i |X_ij-X_ik| <= 1`. Equality is attainable
with all-one versus all-zero columns and all-positive signs.

For `B` independent paired draws, Hoeffding's bounded-mean inequality yields:

```text
u_MC = sqrt(log(1/eta)/(8 B))
Rhat <= min(1/2, mean(Z) + u_MC), with conditional probability >= 1-eta.
```

This is a conservative upper correction, not the Monte Carlo standard error.
There are `B` independent pairs, not `2B` independent observations; the two signs
inside each pair are deliberately dependent.
The latter is reported separately as descriptive numerical precision. The bounded
concentration principle is also given through McDiarmid's inequality in
[Bartlett–Mendelson, Theorem 9](https://www.jmlr.org/papers/volume3/bartlett02a/bartlett02a.pdf).
When all empirical columns are identical, paired scores are identically zero;
Rhat is exactly zero and no Monte Carlo correction is needed.

The implementation substitutes this upper bound into the lower-mean theorem.
By a union bound over data and Monte Carlo failures, the total failure probability
is at most `delta + eta`. Defaults split `total_failure = .05` into `.025 + .025`. For other representable
values, `delta = alpha/2` and `eta = alpha-delta` conserve the budget even at
subnormal magnitudes; inputs too small for a positive split are rejected.
Monte Carlo randomness is independent of the payoff sample and of other replications.

## Simple comparator and fair interpretation

For the same complete fixed family, one-sided Hoeffding and a union over `K`
functions give the deterministic width
`sqrt(log(K/alpha)/(2n))`, at total failure budget `alpha=.05`.
The one-sided inequality and finite-family union argument appear in
[Mohri's finite-family lecture, pages 24–26](https://cs.nyu.edu/~mohri/mls/ml_learning_with_finite_hypothesis_sets.pdf).
Our one-sided construction does not use the extra factor two for a two-sided event.

Both methods therefore receive the **same total failure budget**. Neither is assumed
to be tighter. Their guarantees are separate: selecting the smaller width after
seeing the data does not automatically give a new `.05`-failure procedure. The lab
reports each unchanged and preserves paired width differences across replications.
Bounds are not clipped at zero; uninformative negative lower bounds stay visible.

## Prespecified synthetic protocol

Run without the native OpenQuant extension:

```bash
OPENBLAS_NUM_THREADS=1 python experiments/bounded_finite_family_lab.py \
  --out /tmp/bounded-finite-family.json
python -m pytest python/tests/test_bounded_finite_family_lab.py -q
```

Defaults: 200 replications, 256 rows, 8 candidates, 512 paired sign draws,
seed 7, and all five families. These are bounded mathematical payoffs, not returns
from trading portfolios. Column functions and known means are fixed before sampling.

| Family | Construction | Nominal guarantee applicable? |
| --- | --- | --- |
| `iid_null` | Independent Bernoulli(.5) coordinates and rows | Yes |
| `duplicated_null` | Independent Bernoulli(.5) columns copied in pairs; odd last column unpaired | Yes; duplicates do not enlarge empirical complexity |
| `correlated_null` | Each independent row uses one common Bernoulli(.5) across all columns with probability .75, otherwise independent Bernoulli(.5) values | Yes; cross-candidate dependence is permitted |
| `known_signal` | Column zero has mean .8, the remaining independent Bernoulli columns mean .5 | Yes |
| `repeated_row` | One independent Bernoulli(.5) vector is repeated at every time row | **No**; intentionally violates observation independence |

The dependent control still has unconditional mean .5 in every column. Repeating
one observation does not create more information. Its apparent coverage failure
demonstrates misuse of the IID bound, not a counterexample to the theorem.

Each report retains every replication: input hash, candidate sample/true means,
sample winner, complexity estimate/upper correction, both full lower-bound vectors,
widths, and simultaneous and winner coverage. It also reports whether the selected
lower bound exceeds the synthetic null mean `.5` and whether all lower bounds are
nonpositive. Selection uses the first column on exact ties.

Summary proportions have Wilson 95% intervals over independent simulation
replications. These quantify simulation uncertainty, not the probability a strategy
is profitable. Theoretical budgets apply to each experiment under its assumptions;
they do not promise exact empirical 95% coverage or joint coverage over all settings.
Coverage can be much higher because bounds are conservative. In the known-signal
family, the above-.5 event is a detection diagnostic rather than a false-positive rate.

Input hashes include shape and canonical little-endian float64 bytes in replication
order. Reports include all configuration, source hash, Git base and Python/NumPy
versions. Family-indexed deterministic streams preserve results when requested
family order changes. Reproducibility across numerical-library versions is not promised.

## Acceptance and limits

### Recorded findings, November 7, 2026

The private final protocol retained 400 replications for every family at each
`(rows, candidates, paired draws)` setting: `(128,1,128)`, `(128,8,512)`,
`(256,8,512)`, `(1024,32,1024)` and `(256,8,2)`. Source identity is
`351b05273faed1d9588b349c1ce6ca8a2777dd16a3f1e9ea46194c949570db9b`.
The two 256-row/8-candidate settings have identical payoff input hashes, so changing
the Monte Carlo budget does not change their observed data.

At 1024 rows/32 candidates/1024 draws, mean widths were:

| Family | Corrected Rademacher width | Finite-family union width |
| --- | ---: | ---: |
| IID null | 0.24586 | 0.05617 |
| Duplicated null | 0.23640 | 0.05617 |
| Correlated null | 0.21351 | 0.05617 |
| Known signal | 0.24587 | 0.05617 |

The complexity calculation responded to duplicate/correlated candidates, but the
Rademacher bound was substantially **wider**, not superior, in these controls.
Both methods covered all candidate means in all 400 known-signal replications;
the finite-family method had one simultaneous coverage failure in each of the IID
and duplicated null controls. The Rademacher method had none in the IID controls.
These frequencies are not proof of calibration; their Wilson intervals remain in
the full reports.

The dependence violation was visible at 256 and 1024 rows: both methods had only
1% and 0% simultaneous coverage, respectively, in the repeated-row control.
It was partly **hidden by conservatism** at 128 rows: Rademacher coverage was
99.25% while finite-family coverage was 1%. Passing such an empirical coverage check
therefore does not establish independence or validate a theorem's applicability.

At 256 rows/8 candidates, reducing paired draws from 512 to 2 increased the IID-null
Rademacher mean width from 0.42648 to 1.27753. This is intentionally preserved as a
vacuous numerical-budget outcome. At 256 rows, the corrected Rademacher lower bound
never exceeded `.5` in the known-signal control, while the simpler bound did in all
400 replications. At 1024 rows both did in all 400. No run configuration was selected
as a trading recommendation, and the lower-power result was not tuned away.

Exact enumeration is restricted to at most 16 observations. Tests compare it with
an independently written, unpaired sum over all signs and a hand-calculated two-row
example. Other checks cover range attainability, duplicates, complements, singleton
complexity, Monte Carlo correction constants, fair failure budgets, malformed input,
deterministic replay and a dependent-row counterexample. No empirical significance
test substitutes for verifying the assumptions.

The previous adaptive lab uses unbounded Gaussian/t3 values and constructs a
data-dependent family from validation feedback. Its serial scenario also lacks IID
rows. This theorem cannot be applied to that recorded trajectory by merely computing
its complexity. Clipping payoffs changes the population estimand; normalizing using
the evaluation sample can make the function class data-dependent. Neither is silently
performed here. A family learned from independent training data could instead be
frozen and evaluated conditionally on fresh IID data, but that is a separate protocol.

Continuation: test fixed payoff families with known population means; preserve weak
or vacuous bounds. A dependency-aware RAS implementation requires its complete
primary theorem and assumptions, not inferred constants from secondary code.
