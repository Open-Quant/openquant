"""Mean-variance inputs and reference weights from stock_prices.csv, independent of this library.

    uv run --with pandas --with scipy python tests/fixtures/portfolio_optimization/generate_mean_variance.py
    uv run --with numpy --with scipy python tests/fixtures/portfolio_optimization/generate_qp_reference.py

Writes mean_variance_fixture.json next to this file. Imports neither openquant nor mlfinlab.
qp_reference.json takes its (mu, C) from this file, so run generate_qp_reference.py after it
(the second command above).

Conventions (stated here because they are what the tests compare against):
  returns      simple returns, p_t / p_{t-1} - 1 (pandas pct_change), the convention of
               AFML chapter 16 and of this library's cla, hrp and hcaa modules.
  covariance   sample covariance (ddof=1) of the returns, not annualised.

Blocks:
  weights.inverse_variance   w_i proportional to 1 / C_ii on daily returns.
  weights.min_volatility     argmin w'Cw on daily returns, sum(w) = 1, 0 <= w <= 1, by scipy
                             SLSQP from several starting points (they must agree).
  expected_returns_weekly    252 * mean of weekly returns; weekly prices are the last price of
                             each calendar week (pandas resample("W").last()).
  covariance_weekly          covariance of those weekly returns.

The weights blocks are what `allocate_inverse_variance` / `allocate_min_vol` should produce from
prices under simple returns. The library on main at the time of writing still took log returns
from prices (#110); tests/portfolio_optimization.rs allows for that gap until the fix lands. The
weekly (mu, C) blocks are fixed inputs and involve no returns convention on the library side.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).parent
prices = pd.read_csv(HERE / "stock_prices.csv", index_col=0, parse_dates=[0])
assert not prices.isna().any().any()

daily = prices.pct_change().iloc[1:]
cov_daily = daily.cov().to_numpy()
n = cov_daily.shape[0]

inv = 1.0 / np.diag(cov_daily)
inverse_variance = inv / inv.sum()

scale = np.trace(cov_daily) / n  # objective scaling only; does not move the minimiser
rng = np.random.default_rng(0)
runs = [minimize(lambda w: w @ cov_daily @ w / scale, w0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                 options={"ftol": 1e-16, "maxiter": 5000})
        for w0 in [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(11)]]
runs = [r for r in runs if abs(r.x.sum() - 1) < 1e-9 and (r.x >= -1e-9).all()]
best = min(runs, key=lambda r: r.fun)
spread = max(np.abs(r.x - best.x).max() for r in runs)
assert spread < 1e-5, spread  # a convex problem: every start must land on the same weights
min_volatility = np.clip(best.x, 0.0, None)
min_volatility /= min_volatility.sum()

weekly = prices.resample("W").last().pct_change().iloc[1:]
expected_returns_weekly = 252.0 * weekly.mean().to_numpy()
covariance_weekly = weekly.cov().to_numpy()

out = {
    "source": "tests/fixtures/portfolio_optimization/generate_mean_variance.py: pandas %s, "
              "scipy SLSQP, on stock_prices.csv" % pd.__version__,
    "assets": list(prices.columns),
    "weights": {
        "inverse_variance": [float(x) for x in inverse_variance],
        "min_volatility": [float(x) for x in min_volatility],
    },
    "expected_returns_weekly": [float(x) for x in expected_returns_weekly],
    "covariance_weekly": [[float(x) for x in row] for row in covariance_weekly],
}
(HERE / "mean_variance_fixture.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"{n} assets, {len(daily)} daily and {len(weekly)} weekly returns; "
      f"min-vol starts agree to {spread:.1e}; min-vol weight on asset 0 {min_volatility[0]:.4f}")
