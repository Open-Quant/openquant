# Fixture provenance: independent regeneration (#138)

This file records the fixtures and inline expected values that #138 regenerated from
independent references. It is written to be folded into `tests/FIXTURES.md` (added by #136):
the first table extends that file's "Generated in this repository from independent references"
table, and the four files it lists come out of that file's "Output of running mlfinlab" table,
which is then empty.

Every generator below:

- imports neither `openquant` nor mlfinlab;
- implements a published formula (AFML chapter and snippet, or the paper cited in its
  docstring) with numpy, scipy, pandas or scikit-learn;
- reads only the BSD-3 mlfinlab v0.8.0 CSVs listed in `tests/FIXTURES.md`, scikit-learn's copy
  of the breast-cancer data, or seeded synthetic data;
- gives the command to rerun it in its docstring (`uv run --with ... python <path>`).

The generators and the files they write are this repository's own work, under its MIT license.
No value in them was copied from mlfinlab code or tests of any version.

## Fixtures

| Fixture | Generator | Inputs | Method |
| --- | --- | --- | --- |
| `filters/events.json` | `filters/generate.py` | `filters/dollar_bar_sample.csv` | CUSUM: AFML snippet 2.4 on log prices; z-score: rolling mean + k rolling std (stated in the docstring) |
| `bet_sizing/prob_dynamic_budget.json` | `bet_sizing/generate_prob_dynamic_budget.py` | hand-written events of `bet_sizing.rs` | AFML snippets 10.1-10.4, section 10.2 (budget) |
| `bet_sizing/reserve_fixture.json` | `bet_sizing/generate_reserve.py` | synthetic, `numpy default_rng(138)` | AFML section 10.2 (reserve); two-Gaussian fit by EM |
| `portfolio_optimization/mean_variance_fixture.json` | `portfolio_optimization/generate_mean_variance.py` | `portfolio_optimization/stock_prices.csv` | simple returns; inverse variance; min variance by scipy SLSQP; weekly (mu, C) |
| `portfolio_optimization/qp_reference.json` | `portfolio_optimization/generate_qp_reference.py` (unchanged, rerun) | (mu, C) from `mean_variance_fixture.json` | scipy SLSQP |
| `backtest_statistics/reference.json` | `backtest_statistics/generate.py` | `backtest_statistics/dollar_bar_sample.csv`, inline inputs | AFML ch. 14 (snippet 14.3, section 14.7), Bailey & Lopez de Prado 2012/2014 |
| `codependence/reference.json` | `codependence/generate.py` | `codependence/random_state_42.csv` | MLAM ch. 3 (snippets 3.1-3.3), Szekely, Rizzo & Bakirov 2007 |
| `microstructural_features/reference.json` | `microstructural_features/generate.py` | `microstructural_features/dollar_bar_sample.csv` | AFML ch. 19 (19.3-19.5) |
| `structural_breaks/reference.json` | `structural_breaks/generate.py` | `structural_breaks/dollar_bar_sample.csv` | AFML ch. 17 (17.3.1, 17.3.2, snippets 17.1-17.4, 17.4.3) |
| `volatility/range_reference.json` | `volatility/generate_range.py` | `backtest_statistics/dollar_bar_sample.csv` | Parkinson 1980, Garman & Klass 1980, Yang & Zhang 2000 |
| `etf_trick/reference.json` | `etf_trick/generate.py` | the five `etf_trick/*.csv` | AFML section 2.4.1 (ETF trick), snippet 2.2 (roll gaps) |
| `labeling/reference.json` | `labeling/generate.py` | `filters/dollar_bar_sample.csv` | AFML snippets 2.4, 3.1-3.5 |
| `onc/breast_cancer_reference.json` | `onc/generate_breast_cancer.py` | `onc/breast_cancer.csv` | MLAM snippets 4.1-4.2 with scikit-learn KMeans; clusters found under every seed |

`portfolio_optimization/mean_variance_fixture.json` keeps only the blocks the tests read
(`weights.inverse_variance`, `weights.min_volatility`, `expected_returns_weekly`,
`covariance_weekly`). The old file's other weights and its `errors` block, which held mlfinlab's
error messages, were not used by any test and are gone.

## Inline expected values

Values that were written into tests and quoted from mlfinlab's test suite now come from one of
the fixtures above, or are derived by hand in a comment next to the assertion.

| Test | Now checked against |
| --- | --- |
| `backtest_statistics.rs`, `test_core_backtest_stats.py` | `backtest_statistics/reference.json`; holding period and drawdowns by hand |
| `codependence.rs`, `test_core_codependence.py` | `codependence/reference.json` |
| `microstructural_features.rs`, `test_core_microstructural.py` | `microstructural_features/reference.json`; entropies of "11100001" by hand |
| `structural_breaks.rs`, `test_core_structural_breaks.py` | `structural_breaks/reference.json` |
| `volatility_features.rs`, `test_core_volatility.py` | `volatility/range_reference.json` |
| `etf_trick.rs`, `futures_roll.rs` | `etf_trick/reference.json` (the two ETF values that remain inline pin the library's own output; see the FINDING test) |
| `labeling.rs` | `labeling/reference.json` |
| `onc.rs`, `test_core_onc.py` | `onc/breast_cancer_reference.json` |
| `hrp.rs`, `hcaa.rs` (leaf order) | `hrp/reference.json` (scipy single linkage) |
| `fast_ewma.rs`, `test_core_fast_ewma.py` | by hand: (21 * 1005 + 19 * 1205) / 40 = 1100 |
| `sampling.rs` | by hand, AFML section 4.5.3's worked example (5/6, 3/4, 1; 6/7; 5/14, 3/14, 6/14) |
| `ef3m.rs`, `test_core_ef3m.py` | expected values computed in the test; the moment vector is the example of Lopez de Prado & Foreman (2014), derivable by hand |

## Rerun after #126

#126 moves `portfolio_optimization` to simple returns. `mean_variance_fixture.json` already
uses simple returns (the convention of AFML chapter 16 and of `cla`, `hrp` and `hcaa`), so nothing
needs regenerating; #126's tightened tolerances (1e-12 inverse variance, 1e-4 min volatility)
were checked against this file. If the returns convention ever changes again:

```bash
uv run --with pandas --with scipy python tests/fixtures/portfolio_optimization/generate_mean_variance.py
uv run --with numpy --with scipy python tests/fixtures/portfolio_optimization/generate_qp_reference.py
```
