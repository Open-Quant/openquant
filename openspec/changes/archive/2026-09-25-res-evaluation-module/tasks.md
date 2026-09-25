## 1. Verify the existing statistics

- [x] 1.1 Check `backtest_stats` PSR, DSR (both trial forms) and MinTRL against a standard-library hand computation on a committed returns series

## 2. Module

- [x] 2.1 Add `python/openquant/evaluation.py` (moments, PSR, DSR, SR0, MinTRL, meta-label metrics, strategy-failure probability)
- [x] 2.2 Add `TrialRegistry` with atomic JSON persistence and registry-deflated DSR
- [x] 2.3 Export `evaluation` from `openquant`

## 3. Tests

- [x] 3.1 Commit `python/tests/fixtures/evaluation_returns.csv` and its generator
- [x] 3.2 Hand-computed PSR, DSR, MinTRL tests; registry persistence across two runs; registry-count DSR; validation

## 4. Docs

- [x] 4.1 Authored page `modules/evaluation.md` with an executed Python example
- [x] 4.2 Register it in `moduleDocs.ts`, the sidebar, the API inventory and the coverage page
