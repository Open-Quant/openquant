# Portfolio optimization fixtures

The fixtures live in `tests/fixtures/portfolio_optimization/` at the repository root:

- `stock_prices.csv`: copied from mlfinlab v0.8.0 `tests/test_data/stock_prices.csv`
  (BSD-3-Clause; see `tests/FIXTURES.md`).
- `mean_variance_fixture.json`: written by `generate_mean_variance.py` (simple returns,
  inverse-variance and minimum-variance weights, weekly expected returns and covariance).
- `qp_reference.json`: written by `generate_qp_reference.py` from the weekly (mu, C) above; run
  it after `generate_mean_variance.py`.

Used for the mean-variance tests (inverse variance, min_volatility, max_sharpe, efficient_risk)
and the CLA reference tests.
