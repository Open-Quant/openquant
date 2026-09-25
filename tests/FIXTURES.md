# Test fixture provenance

Where every file under `tests/fixtures/` came from, under what license, and how to
regenerate it. This answers hypothesis H4 of `docs/design/production-readiness-brief.md`
("fixtures derived from mlfinlab tests have acceptable license provenance") as far as
it can be answered from public sources; the open items are listed at the end.

Last checked: 2026-09-25. This is a record of facts, not legal advice.

## mlfinlab's license history

Many fixtures and several expected values were taken from
[mlfinlab](https://github.com/hudson-and-thames/mlfinlab) (Hudson and Thames
Quantitative Research), whose license changed over time. The upstream repository's
history was rewritten in 2021 (its root commit is now `f71b2bb`, 2021-08-24), so the
early history was read from the fork
[`quantopian/mlfinlab`](https://github.com/quantopian/mlfinlab), made on 2020-07-13,
which still has it:

| Date | Commit (quantopian/mlfinlab) | Version | License |
| --- | --- | --- | --- |
| 2019-05-16 | `a424f8e` | 0.x | BSD 3-Clause, "Copyright (c) 2019, Hudson and Thames Quantitative Research" |
| 2020-03-30 | `59f1457` | 0.8.0 | BSD 3-Clause; `setup.cfg`: `licence = BSD-3` |
| 2020-04-03 | `c504532` | | BSD 3-Clause plus a request to notify the authors of commercial use |
| 2020-04-11 | `95e645a` | | "All rights reserved": use as-is only; reproduction, distribution and derivative works need written permission |
| 2020-07-03 | `8202463` | 0.12.3 | same; `setup.cfg`: `licence = All Rights Reserved` |
| 2021-06 / 2021-11 | upstream `f71b2bb`, `5b6ef4e` | 1.x | proprietary "Copyright Protection Notice and Licensing Agreement" |

mlfinlab is no longer on PyPI (the JSON API returns 404 and the simple index lists no
files), so PyPI metadata cannot be used as a second source.

**Consequence:** material taken from mlfinlab at or before commit `59f1457` (v0.8.0,
2020-03-30) is BSD-3-Clause and may be redistributed here with its copyright notice
(reproduced at the end of this file). Material taken from any later version is not
covered by an open license.

## Fixture by fixture

Git blob hashes (`git ls-files -s`) were compared with the upstream tree at `59f1457`.

### Copied verbatim from mlfinlab `tests/test_data/` (BSD-3-Clause)

Each file is byte-identical (same git blob) to the file of the same name in
`mlfinlab/tests/test_data/` at commit `59f1457` (v0.8.0, BSD-3-Clause).

| File in this repository | Upstream file | Blob |
| --- | --- | --- |
| `backtest_statistics/dollar_bar_sample.csv` | `dollar_bar_sample.csv` | `1a24ee6` |
| `filters/dollar_bar_sample.csv` | `dollar_bar_sample.csv` | `1a24ee6` |
| `microstructural_features/dollar_bar_sample.csv` | `dollar_bar_sample.csv` | `1a24ee6` |
| `structural_breaks/dollar_bar_sample.csv` | `dollar_bar_sample.csv` | `1a24ee6` |
| `microstructural_features/tick_data.csv` | `tick_data.csv` | `3a92250` |
| `microstructural_features/tick_data_time_bars.csv` | `tick_data_time_bars.csv` | `4b75f8f` |
| `portfolio_optimization/stock_prices.csv` | `stock_prices.csv` | `ebb6d80` |
| `etf_trick/alloc_df.csv` | `alloc_df.csv` | `0cd20f6` |
| `etf_trick/close_df.csv` | `close_df.csv` | `9766132` |
| `etf_trick/costs_df.csv` | `costs_df.csv` | `2445131` |
| `etf_trick/open_df.csv` | `open_df.csv` | `b2d5324` |
| `etf_trick/rates_df.csv` | `rates_df.csv` | `6a2f9e6` |

mlfinlab does not say where the underlying market data in these files came from
(the ETF prices in `stock_prices.csv`, the tick and bar samples). The BSD grant covers
Hudson and Thames's rights in the files, not any third-party data vendor's.

### Regenerated from a seed (no data copied)

- `codependence/random_state_42.csv`: the three series mlfinlab's
  `test_codependence.py` builds in its `setUp`. Regenerates to within floating-point
  rounding, not bit-for-bit (maximum difference 3.6e-15 with numpy 2.5.3 on arm64), with:

  ```python
  import numpy as np
  state = np.random.RandomState(42)
  x = state.normal(size=1000)
  y_1 = x ** 2 + state.normal(size=1000) / 5
  y_2 = abs(x) + state.normal(size=1000) / 5
  ```

### Third-party data, not from mlfinlab

- `onc/breast_cancer.csv`: byte-identical (blob `979a3dc`) to scikit-learn's
  `sklearn/datasets/data/breast_cancer.csv` (scikit-learn is BSD-3-Clause). The data is
  the UCI Breast Cancer Wisconsin (Diagnostic) dataset, licensed CC BY 4.0: Wolberg,
  Mangasarian, Street and Street (1993), UCI Machine Learning Repository,
  <https://doi.org/10.24432/C5DW2B>.

### Generated in this repository from independent references

Each has a generator beside it that does not import `openquant` or mlfinlab. They
implement a published formula (AFML chapter and snippet, or the paper cited in the
docstring) with numpy, scipy, pandas or scikit-learn, and read only the BSD-3 mlfinlab
v0.8.0 CSVs listed above, scikit-learn's copy of the breast-cancer data, or seeded
synthetic data. The docstring of each script gives the source and the command to rerun it
(`uv run --with ... python <path>`). The generators and the files they write are this
repository's own work, under its MIT license; no value in them was copied from mlfinlab
code or tests of any version.

| Fixture | Generator | Inputs | Method |
| --- | --- | --- | --- |
| `feature_importance/pca_reference.json` | `feature_importance/generate.py` | synthetic, seeded | see the docstring |
| `hrp/reference.json` | `hrp/generate.py` | `portfolio_optimization/stock_prices.csv`, seeded covariance | see the docstring |
| `onc/silhouette_reference.json` | `onc/generate.py` | synthetic, seeded | see the docstring |
| `sample_weights/reference.json` | `sample_weights/generate.py` | `filters/dollar_bar_sample.csv` | see the docstring |
| `volatility/daily_vol_reference.json` | `volatility/generate.py` | `filters/dollar_bar_sample.csv` | see the docstring |
| `portfolio_optimization/qp_reference.json` | `portfolio_optimization/generate_qp_reference.py` | `expected_returns_weekly` and `covariance_weekly` from `mean_variance_fixture.json` | scipy SLSQP |
| `filters/events.json` | `filters/generate.py` | `filters/dollar_bar_sample.csv` | CUSUM: AFML snippet 2.4 on log prices; z-score: rolling mean + k rolling std (stated in the docstring) |
| `bet_sizing/prob_dynamic_budget.json` | `bet_sizing/generate_prob_dynamic_budget.py` | hand-written events of `bet_sizing.rs` | AFML snippets 10.1-10.4, section 10.2 (budget) |
| `bet_sizing/reserve_fixture.json` | `bet_sizing/generate_reserve.py` | synthetic, `numpy default_rng(138)` | AFML section 10.2 (reserve); two-Gaussian fit by EM |
| `portfolio_optimization/mean_variance_fixture.json` | `portfolio_optimization/generate_mean_variance.py` | `portfolio_optimization/stock_prices.csv` | simple returns; inverse variance; min variance by scipy SLSQP; weekly (mu, C) |
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
`covariance_weekly`). The old file's other weights and its `errors` block, which held
mlfinlab's error messages, were not used by any test and are gone. It uses simple returns
(the convention of AFML chapter 16, of #126's `portfolio_optimization`, and of `cla`, `hrp`
and `hcaa`). If the returns convention ever changes:

```bash
uv run --with pandas --with scipy python tests/fixtures/portfolio_optimization/generate_mean_variance.py
uv run --with numpy --with scipy python tests/fixtures/portfolio_optimization/generate_qp_reference.py
```

### Output of running mlfinlab: provenance not established

None. The four files that used to be listed here (`filters/events.json`,
`bet_sizing/reserve_fixture.json`, `bet_sizing/prob_dynamic_budget.json`,
`portfolio_optimization/mean_variance_fixture.json`) were regenerated from independent
references by #138 and are listed in the table above.

### Inline expected values

Values that were written into tests and quoted from mlfinlab's test suite now come from one
of the fixtures above, or are derived by hand in a comment next to the assertion.

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

## Open items

1. Decide whether the BSD notice below should also ship in any source distribution that
   includes `tests/fixtures/` (it does today only because this file is in the repository).

## mlfinlab BSD 3-Clause notice

Applies to the files listed under "Copied verbatim from mlfinlab" above, as released at
mlfinlab commit `59f1457` (v0.8.0).

```text
BSD 3-Clause License

Copyright (c) 2019, Hudson and Thames Quantitative Research
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
