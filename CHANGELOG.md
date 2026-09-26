# Changelog

Notable changes to OpenQuant, newest first within each section. OpenQuant is pre-release:
the crates and the Python package are at 0.1.0, no version has been tagged and nothing is
published to crates.io or PyPI, so every change so far sits under **Unreleased**. When a
version is tagged, its entries move under a heading with that version and date.

Each entry links the pull request that made the change; the linked issue, where there is
one, is in parentheses. Entries were seeded from the titles of the merged pull requests.
From now on, a pull request with a user-visible effect adds one line to the right section
below, in the same style as its title. The docs site renders this file as its Changelog
page (`python3 scripts/docs/generate_site_pages.py --write`; CI fails if the page and this
file disagree).

## Unreleased

### Added

- Docs: a research gallery generated from the executed runbooks, a performance page generated from `benchmarks/`, and this changelog on the site; the governance stubs folded into one page (#55) ([#207](https://github.com/Open-Quant/openquant/pull/207))
- `hcaa`: linkage option (single, complete, average, ward), Ward by default (#197) ([#203](https://github.com/Open-Quant/openquant/pull/203))
- Runbooks 11 and 13: a sequential-bootstrap bagging comparison, and the workarounds for #187 removed ([#199](https://github.com/Open-Quant/openquant/pull/199))
- `hcaa`: `distance` option shared with HRP, pairwise default kept (#188) ([#195](https://github.com/Open-Quant/openquant/pull/195))
- Runbook 13, bet sizing from predicted probabilities (#50) ([#191](https://github.com/Open-Quant/openquant/pull/191))
- Runbook 12, CPCV backtest with PSR and the deflated Sharpe ratio (#48) ([#179](https://github.com/Open-Quant/openquant/pull/179))
- Runbook 11, triple-barrier labeling and meta-labeling (#47) ([#181](https://github.com/Open-Quant/openquant/pull/181))
- Runbook 10, HRP against inverse-variance and CLA out of sample (#51) ([#149](https://github.com/Open-Quant/openquant/pull/149))
- Runbook 09, fracdiff stationarity versus memory (#49) ([#148](https://github.com/Open-Quant/openquant/pull/148))
- Notebooks executed with nbclient, committed with outputs and figures, and run in CI (#45) ([#146](https://github.com/Open-Quant/openquant/pull/146))
- Python bindings for purged CV, CPCV backtesting, feature importance and tuning (#42) ([#143](https://github.com/Open-Quant/openquant/pull/143))
- `openquant.data.fetch` with a Parquet cache, the dataset hash in run manifests, and `DATA_SOURCES.md` (#43) ([#141](https://github.com/Open-Quant/openquant/pull/141))
- `openquant.evaluation`: PSR, DSR, MinTRL and a trial registry (#44) ([#139](https://github.com/Open-Quant/openquant/pull/139))
- Recovered stranded work: purged-CV diagnostics, CPCV and pipeline plots re-ported, outcomes recorded (#33) ([#137](https://github.com/Open-Quant/openquant/pull/137))
- AFML Chapter 21 pigeonhole-partition trajectory search (#112) ([#131](https://github.com/Open-Quant/openquant/pull/131))
- The Critical Line Algorithm; the `cla` module was a stub (#76) ([#83](https://github.com/Open-Quant/openquant/pull/83))
- `streaming_hpc` analytics utilities, AFML Chapter 22 ([#12](https://github.com/Open-Quant/openquant/pull/12))
- Combinatorial optimization adapters, AFML Chapter 21 ([#11](https://github.com/Open-Quant/openquant/pull/11))
- `hpc_parallel` utilities, AFML Chapter 20 ([#10](https://github.com/Open-Quant/openquant/pull/10))
- Typed error APIs for `bet_sizing` and `filters` ([#9](https://github.com/Open-Quant/openquant/pull/9))
- Strategy risk diagnostics, AFML Chapter 15 ([#8](https://github.com/Open-Quant/openquant/pull/8))
- Synthetic backtesting, AFML Chapter 13 ([#7](https://github.com/Open-Quant/openquant/pull/7))
- Backtesting engine and CPCV, AFML Chapters 11–12 ([#6](https://github.com/Open-Quant/openquant/pull/6))
- `hyperparameter_tuning`, AFML Chapter 9 ([#5](https://github.com/Open-Quant/openquant/pull/5))
- `ensemble_methods`, AFML Chapter 6 ([#4](https://github.com/Open-Quant/openquant/pull/4))
- AFML notebook platform and a real-data end-to-end research workflow ([#3](https://github.com/Open-Quant/openquant/pull/3))
- Notebook-first AFML research platform: bindings, pipeline, notebooks, CI ([#2](https://github.com/Open-Quant/openquant/pull/2))

### Fixed

- A binding default that always failed, ignored parameters and silent drops (#194) ([#202](https://github.com/Open-Quant/openquant/pull/202))
- Input validation and minor cleanups from the rustdoc audit (#186) ([#204](https://github.com/Open-Quant/openquant/pull/204))
- Docs/code mismatches in `pipeline`, `hcaa`, `onc`, `cross_validation`, `combinatorial_optimization`, `microstructural_features` and `streaming_hpc` (#185) ([#201](https://github.com/Open-Quant/openquant/pull/201))
- `sb_bagging` rejects a wrong column count and wraps the warm-start seed (#184) ([#196](https://github.com/Open-Quant/openquant/pull/196))
- Typed errors instead of reachable panics, overflows and infinite loops (#184) ([#193](https://github.com/Open-Quant/openquant/pull/193))
- Python API gaps found by the meta-labeling runbook (#187) ([#190](https://github.com/Open-Quant/openquant/pull/190))
- `cla` error variants and panic guard, `data_processing` off-grid report and gap rule, `strategy_risk` unreachable targets (#168) ([#189](https://github.com/Open-Quant/openquant/pull/189))
- `hrp` clusters on AFML's distance of distances by default (#167) ([#183](https://github.com/Open-Quant/openquant/pull/183))
- Chu-Stinchcombe-White: σ²ₜ is the mean of the squared differences (#173) ([#182](https://github.com/Open-Quant/openquant/pull/182))
- `get_sadf` quadratic and sub/super-martingale models follow AFML §17.4 (#166) ([#178](https://github.com/Open-Quant/openquant/pull/178))
- `get_yang_zhang_vol` implements Yang and Zhang (2000) (#165) ([#176](https://github.com/Open-Quant/openquant/pull/176))
- An event on the last bar with no vertical barrier keeps `t1 = None` (#162) ([#175](https://github.com/Open-Quant/openquant/pull/175))
- ETF-trick holdings are sized from the previous bar, per AFML §2.4.1 (#164) ([#174](https://github.com/Open-Quant/openquant/pull/174))
- `bet_sizing` limit price defined for reducing, short and sign-flipping moves (#163) ([#172](https://github.com/Open-Quant/openquant/pull/172))
- `get_weights_ffd` honours `lim` for every value, so `thresh <= 0` cannot hang (#169) ([#171](https://github.com/Open-Quant/openquant/pull/171))
- `PurgedKFold` embargoes only after each test block, from the end of the purge (#134) ([#170](https://github.com/Open-Quant/openquant/pull/170))
- Python binding ergonomics before the API is published (#77) ([#145](https://github.com/Open-Quant/openquant/pull/145))
- `bias_variance_noise` reports real noise only given the noiseless target (#128) ([#140](https://github.com/Open-Quant/openquant/pull/140))
- `ef3m` five-moment variant fits about the mean, so μ₂ can be negative (#115) ([#135](https://github.com/Open-Quant/openquant/pull/135))
- `streaming_hpc` alerts on the CDF of VPIN and weights HHI by volume (#113) ([#133](https://github.com/Open-Quant/openquant/pull/133))
- `feature_importance` and `backtesting_engine` match their AFML references (#94) ([#132](https://github.com/Open-Quant/openquant/pull/132))
- Python bindings keep sub-second timestamps (#87) ([#129](https://github.com/Open-Quant/openquant/pull/129))
- MDA shuffles the test-fold column instead of rotating it (#98) ([#127](https://github.com/Open-Quant/openquant/pull/127))
- `portfolio_optimization` uses simple returns and reports annual risk and Sharpe ratio (#110) ([#126](https://github.com/Open-Quant/openquant/pull/126))
- `sb_bagging` draws each estimator's sample with the sequential bootstrap (#90) ([#125](https://github.com/Open-Quant/openquant/pull/125))
- `feature_diagnostics` fits a real logistic regression (#99) ([#124](https://github.com/Open-Quant/openquant/pull/124))
- Time-decay weights keep one weight per event when starts coincide (#91) ([#122](https://github.com/Open-Quant/openquant/pull/122))
- Conditional drawdown at risk averages the drawdown tail (#102) ([#121](https://github.com/Open-Quant/openquant/pull/121))
- Chu-Stinchcombe-White statistic divides by the standard deviation (#104) ([#120](https://github.com/Open-Quant/openquant/pull/120))
- Hasbrouck's lambda regresses the signed return (#105) ([#119](https://github.com/Open-Quant/openquant/pull/119))
- HCAA allocates down the tree and honours `optimal_num_clusters` (#108) ([#118](https://github.com/Open-Quant/openquant/pull/118))
- ONC keeps the re-clustered partition when it scores higher (#107) ([#117](https://github.com/Open-Quant/openquant/pull/117))
- `ef3m` fit rows report their own error; `fit_m2n` honours `n_runs` ([#116](https://github.com/Open-Quant/openquant/pull/116))
- Nested partitions and the trajectory path cap in the Chapter 20–22 modules ([#114](https://github.com/Open-Quant/openquant/pull/114))
- `efficient_risk` failed on feasible targets ([#111](https://github.com/Open-Quant/openquant/pull/111))
- Weekly and monthly resampling scrambled the price matrix in four modules (#93) ([#96](https://github.com/Open-Quant/openquant/pull/96))
- Invalid arguments return `Err` instead of panicking (#36) ([#84](https://github.com/Open-Quant/openquant/pull/84))
- The constrained mean-variance problems are solved instead of clamping the unconstrained ones ([#82](https://github.com/Open-Quant/openquant/pull/82))
- Daily volatility, event resolution and return attribution match AFML ([#81](https://github.com/Open-Quant/openquant/pull/81))
- Hard-coded HRP, HCAA and ONC answers removed, and the ONC bugs they hid fixed (#75) ([#80](https://github.com/Open-Quant/openquant/pull/80))
- Python matrices are read as rows of observations, not column-major (#74) ([#79](https://github.com/Open-Quant/openquant/pull/79))
- `PurgedKFold::split` leaked: the test window now starts at the first test label's end time (#65) ([#66](https://github.com/Open-Quant/openquant/pull/66))
- The first tick threshold is consumed, so features align with bars (#67) ([#68](https://github.com/Open-Quant/openquant/pull/68))
- Unpurged feature-importance CV can no longer happen silently ([#27](https://github.com/Open-Quant/openquant/pull/27))
- `get_parkinson_vol` spelled correctly in the public API ([#24](https://github.com/Open-Quant/openquant/pull/24))

### Changed

- The docs site and README use the Monograph identity (#57) ([#86](https://github.com/Open-Quant/openquant/pull/86))
- Typed errors across the Rust core; the bindings raise the core's message (#35) ([#85](https://github.com/Open-Quant/openquant/pull/85))
- One visual identity: palette, type, logo (#56) ([#72](https://github.com/Open-Quant/openquant/pull/72))
- The Python distribution is named `pyopenquant`, with accurate metadata ([#70](https://github.com/Open-Quant/openquant/pull/70))
- AFML book-scraping tooling removed (#32) ([#69](https://github.com/Open-Quant/openquant/pull/69))
- LazyFrame-native `openquant.data` path and a benchmark scaffold ([#15](https://github.com/Open-Quant/openquant/pull/15))

### Documentation

- Rustdoc under `/api/rust/`, a generated Python reference, docstrings and `.pyi` stubs (#54) ([#192](https://github.com/Open-Quant/openquant/pull/192))
- Rustdoc and doctests for the remaining `openquant` modules (#37) ([#180](https://github.com/Open-Quant/openquant/pull/180))
- Rustdoc and doctests for 15 `openquant` modules (#37) ([#144](https://github.com/Open-Quant/openquant/pull/144))
- Contributor and community files; test fixture provenance recorded (#61) ([#136](https://github.com/Open-Quant/openquant/pull/136))
- The vendored pyo3-polars patch documented and guarded with a test (#59) ([#142](https://github.com/Open-Quant/openquant/pull/142))
- Hand-written `hrp`, `hcaa` and `onc` pages; a claim on the `risk_metrics` page corrected ([#109](https://github.com/Open-Quant/openquant/pull/109))
- Hand-written `structural_breaks` and `microstructural_features` pages ([#106](https://github.com/Open-Quant/openquant/pull/106))
- `risk_metrics`, `strategy_risk` and `codependence` pages; a `codependence` panic fixed ([#103](https://github.com/Open-Quant/openquant/pull/103))
- Hand-written backtesting pages (engine, statistics, synthetic) and a phone-width math fix ([#101](https://github.com/Open-Quant/openquant/pull/101))
- Hand-written Chapter 8–9 module pages (feature importance, diagnostics, fingerprint, tuning) ([#100](https://github.com/Open-Quant/openquant/pull/100))
- Hand-written Chapter 5–7 module pages (`fracdiff`, `ensemble_methods`, `cross_validation`) ([#97](https://github.com/Open-Quant/openquant/pull/97))
- Hand-written Chapter 4 module pages (`sampling`, `sample_weights`, `sb_bagging`) ([#92](https://github.com/Open-Quant/openquant/pull/92))
- Hand-written `labeling` and `bet_sizing` pages ([#89](https://github.com/Open-Quant/openquant/pull/89))
- Hand-written Chapter 2 module pages, the `authored` status and the citation gate ([#88](https://github.com/Open-Quant/openquant/pull/88))
- The two serious known defects named on the landing page ([#78](https://github.com/Open-Quant/openquant/pull/78))
- The landing page describes the library, not the documentation (#52) ([#71](https://github.com/Open-Quant/openquant/pull/71))
- The coverage dashboard recomputes itself ([#30](https://github.com/Open-Quant/openquant/pull/30))
- `check:examples` wired into CI, the status badge derived, the information architecture consolidated ([#26](https://github.com/Open-Quant/openquant/pull/26))
- The `get_parkinson_vol` rename propagated into the docs ([#25](https://github.com/Open-Quant/openquant/pull/25))
- The eight setup and workflow pages deliver on their titles ([#22](https://github.com/Open-Quant/openquant/pull/22))
- Every module page says something true and compile-checked ([#21](https://github.com/Open-Quant/openquant/pull/21))
- The API-drift gate sees the whole Rust surface ([#19](https://github.com/Open-Quant/openquant/pull/19))
- The site no longer claims a review that never happened ([#18](https://github.com/Open-Quant/openquant/pull/18))
- Docs site: 13 production redirects fixed, doc gates wired into CI, sitemap emitted ([#17](https://github.com/Open-Quant/openquant/pull/17))
- Notebook-first workflow page on the docs site ([#14](https://github.com/Open-Quant/openquant/pull/14))

### Tests

- Fixtures and inline values regenerated without mlfinlab provenance (#138) ([#147](https://github.com/Open-Quant/openquant/pull/147))
- Measured test-sensitivity audit and reference tests for eight modules (#40) ([#95](https://github.com/Open-Quant/openquant/pull/95))
- Every Python submodule covered and exposed from the package (#39) ([#73](https://github.com/Open-Quant/openquant/pull/73))

### CI, build and tooling

- The workspace is migrated to Rust edition 2024 ([#200](https://github.com/Open-Quant/openquant/pull/200))
- Rust caching, parallel lint and test jobs, release-mode parallel notebooks, a docs-only skip ([#198](https://github.com/Open-Quant/openquant/pull/198))
- CI gates what it claims to gate (#41) ([#130](https://github.com/Open-Quant/openquant/pull/130))
- Remaining local tooling ignored and the last stale doc lines fixed ([#123](https://github.com/Open-Quant/openquant/pull/123))
- The workspace is clippy-clean under `-D warnings` (#34) ([#64](https://github.com/Open-Quant/openquant/pull/64))
- Repository root cleaned, local tooling ignored, merged branches pruned (#31) ([#63](https://github.com/Open-Quant/openquant/pull/63))
- AI-DLC adopted and the production-readiness backlog recorded ([#62](https://github.com/Open-Quant/openquant/pull/62))
- The documented Python examples run in CI, and the three that were broken fixed ([#29](https://github.com/Open-Quant/openquant/pull/29))
- `ethnum` bumped from 1.5.2 to 1.5.3 so the crate compiles on stable ([#28](https://github.com/Open-Quant/openquant/pull/28))
- `cargo fmt` applied to `pyopenquant` ([#20](https://github.com/Open-Quant/openquant/pull/20))
- Per-function data-processing benchmark metrics ([#16](https://github.com/Open-Quant/openquant/pull/16))
- Beads closure history reconciled with the delivered work ([#13](https://github.com/Open-Quant/openquant/pull/13))

### Dependencies

- The cargo minor/patch group, 4 updates ([#154](https://github.com/Open-Quant/openquant/pull/154))
- The Python minor/patch group, 2 updates ([#152](https://github.com/Open-Quant/openquant/pull/152))
