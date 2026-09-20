"""Independent reference for the mean-variance solvers: scipy on a fixed (mu, C).

    uv run --with numpy --with scipy python tests/fixtures/portfolio_optimization/generate_qp_reference.py

Inputs are `expected_returns_weekly` and `covariance_weekly` from mean_variance_fixture.json,
so no returns convention is involved: this checks the optimisers and nothing else. Max Sharpe
is maximised directly (not through the homogenising substitution the library uses), from
several starting points, so it is an independent check of that transformation too.

The older `weights` block in mean_variance_fixture.json is not used: its own `errors` block
records that the generator failed for three solutions, and its `*_bound0` cases do not
correspond to the bounds the Rust tests apply.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).parent
fixture = json.loads((HERE / "mean_variance_fixture.json").read_text())
mu = np.array(fixture["expected_returns_weekly"], dtype=float).ravel()
C = np.array(fixture["covariance_weekly"], dtype=float)
n = len(mu)
scale = np.trace(C) / n  # objective scaling only; does not move the minimiser
OPTS = {"ftol": 1e-16, "maxiter": 5000}


def starts(bounds):
    rng = np.random.default_rng(0)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    out = []
    for _ in range(12):
        w = np.clip(rng.dirichlet(np.ones(n)), lo, hi)
        out.append(w / w.sum())
    return out


def best(objective, bounds, constraints):
    results = [minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options=OPTS)
               for w0 in starts(bounds)]
    # SLSQP often reports "positive directional derivative" at this ftol although it has
    # converged, so results are judged by feasibility and objective, not by the success flag.
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    def feasible(w):
        return (abs(w.sum() - 1) < 1e-9 and (w >= lo - 1e-9).all() and (w <= hi + 1e-9).all()
                and all(c["fun"](w) > -1e-9 for c in constraints if c["type"] == "ineq"))

    ok = [r for r in results if np.isfinite(r.fun) and feasible(r.x)]
    assert ok, "no feasible result"
    values = sorted(r.fun for r in ok)
    # Convex problems (and max Sharpe, which is quasi-convex) have one optimum: the starts
    # must agree, or this reference is not trustworthy.
    assert values[len(values) // 2] - values[0] < 1e-9 * max(1.0, abs(values[0])), values[:4]
    return min(ok, key=lambda r: r.fun).x


budget = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
variance = lambda w: w @ C @ w / scale
neg_sharpe = lambda w: -(w @ mu) / np.sqrt(w @ C @ w)

long_only = [(0.0, 1.0)] * n
asset0_floor = [(0.3, 1.0)] + [(0.0, 1.0)] * (n - 1)
capped = [(0.01, 0.15)] * n

w_min = best(variance, long_only, [budget])
target = float(0.5 * (w_min @ mu + mu.max()))  # strictly between min-variance return and the best asset

cases = {
    "min_vol": {"bounds": "long_only", "weights": best(variance, long_only, [budget])},
    "min_vol_asset0_floor": {"bounds": "asset0>=0.3", "weights": best(variance, asset0_floor, [budget])},
    "min_vol_capped": {"bounds": "all in [0.01, 0.15]", "weights": best(variance, capped, [budget])},
    "max_sharpe": {"bounds": "long_only", "risk_free": 0.0, "weights": best(neg_sharpe, long_only, [budget])},
    "max_sharpe_asset0_floor": {"bounds": "asset0>=0.3", "risk_free": 0.0, "weights": best(neg_sharpe, asset0_floor, [budget])},
    "efficient_risk": {"bounds": "long_only", "target_return": target,
                       "weights": best(variance, long_only, [budget, {"type": "ineq", "fun": lambda w: w @ mu - target}])},
    "efficient_risk_below_min_var": {"bounds": "long_only", "target_return": float(w_min @ mu - 1e-3),
                                     "weights": best(variance, long_only, [budget, {"type": "ineq", "fun": lambda w: w @ mu - (w_min @ mu - 1e-3)}])},
}
for name, case in cases.items():
    w = case["weights"]
    case["variance"] = float(w @ C @ w)
    case["return"] = float(w @ mu)
    case["weights"] = [float(x) for x in w]
    print(f"{name:30s} var={case['variance']:.3e} ret={case['return']:+.5f} nonzero={int((w > 1e-7).sum()):2d} max={w.max():.4f}")

(HERE / "qp_reference.json").write_text(json.dumps({"source": __doc__.strip().splitlines()[0], "cases": cases}, indent=2) + "\n")
print("positive mu:", int((mu > 0).sum()), "of", n)
