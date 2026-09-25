# Test fixture provenance

Where every file under `tests/fixtures/` came from, under what license, and how to
regenerate it. This answers hypothesis H4 of `docs/design/production-readiness-brief.md`
("fixtures derived from mlfinlab tests have acceptable license provenance") as far as
it can be answered from public sources; the open items are listed at the end.

Last checked: 2026-09-24. This is a record of facts, not legal advice.

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
  `test_codependence.py` builds in its `setUp`. Regenerates exactly (maximum difference
  0.0) with:

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

Each has a `generate.py` beside it that does not import `openquant` or mlfinlab. They
implement AFML snippets or call numpy, scipy, pandas or scikit-learn; the docstring of
each script gives the source and the command.

| Fixture | Generator | Inputs |
| --- | --- | --- |
| `feature_importance/pca_reference.json` | `feature_importance/generate.py` | synthetic, seeded |
| `hrp/reference.json` | `hrp/generate.py` | `portfolio_optimization/stock_prices.csv`, seeded covariance |
| `onc/silhouette_reference.json` | `onc/generate.py` | synthetic, seeded |
| `sample_weights/reference.json` | `sample_weights/generate.py` | `filters/dollar_bar_sample.csv` |
| `volatility/daily_vol_reference.json` | `volatility/generate.py` | `filters/dollar_bar_sample.csv` |
| `portfolio_optimization/qp_reference.json` | `portfolio_optimization/generate_qp_reference.py` | `expected_returns_weekly` and `covariance_weekly` from `mean_variance_fixture.json` (see below) |

### Output of running mlfinlab: provenance not established

These files hold values computed by running mlfinlab code. `docs/python_pytest_baseline.md`
records that the reference environment was a local mlfinlab checkout described as
"v1.0", with local modifications; v1.0 postdates the April 2020 license change (v0.12.3
was already "All Rights Reserved"), and the exact revision and its license were not
recorded. The generators for some of them are not in this repository.

| File | What it is | Generator |
| --- | --- | --- |
| `filters/events.json` | `cusum_filter` / `z_score_filter` output on `dollar_bar_sample.csv` | inline script in `filters/README.md` (imports mlfinlab) |
| `bet_sizing/reserve_fixture.json` | EF3M fit and `bet_size` on a 500-sample synthetic set | `scripts/gen_bet_sizing_fixtures.py` in the mlfinlab checkout, not in this repository (`crates/openquant/tests/fixtures/bet_sizing/README.md`) |
| `bet_sizing/prob_dynamic_budget.json` | bet-sizing outputs (probability, dynamic, budget) | not recorded |
| `portfolio_optimization/mean_variance_fixture.json` | mean-variance weights, plus weekly expected returns and covariance of `stock_prices.csv`; its `errors` block holds mlfinlab's own error message | not recorded |

Computed numbers are not obviously copyrightable, but whether they may be kept, and
whether they should be replaced with independent references (as was done for
`sample_weights`, `volatility` and `hrp`), is for the maintainer to decide.

The same applies to expected values written inline in tests that were quoted from
mlfinlab's test suite (see `tests/crosswalk.md` and `docs/test-sensitivity-audit.md`,
"mlfinlab reference literals"): which mlfinlab version they were copied from is not
recorded.

## Open items

1. Record, or recover, the mlfinlab revision used for the four "output of running
   mlfinlab" files above, or regenerate them from independent references.
2. Record which mlfinlab version the inline expected values were quoted from.
3. Decide whether the BSD notice below should also ship in any source distribution that
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
