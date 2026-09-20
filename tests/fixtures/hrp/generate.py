"""Independent HRP reference values for crates/openquant/tests/hrp_reference.rs.

Run from the repository root:

    uv run --no-project --with numpy --with scipy python tests/fixtures/hrp/generate.py

(--no-project keeps numpy/scipy out of this repository's own Python environment.)

This is a direct transcription of Lopez de Prado, "Advances in Financial Machine Learning",
snippets 16.1-16.4 (getIVP, getClusterVar, getQuasiDiag, getRecBipart), with the clustering
step done by scipy.  It shares no code with the Rust library.  The one deliberate choice: the
condensed correlation-distance matrix d_ij = sqrt((1 - rho_ij) / 2) is handed to
scipy.cluster.hierarchy.linkage(method="single") as pairwise distances (what mlfinlab does),
not as an observation matrix (what the book's snippet 16.4 does).

Cases written to reference.json:
  * stock_prices: the 23-ETF price fixture, simple returns, sample covariance (ddof=1).
  * stock_prices_weekly: every 5th row of the same prices (rows 4, 9, 14, ...), the rule the
    library documents for resample_by="W".
  * random_cov_8: a seeded random 8x8 covariance passed directly.
  * stock_prices_shrunk: as stock_prices, but every off-diagonal covariance is multiplied by
    0.9 first.  That is the rule the library applies for use_shrinkage=true (constant 10%
    shrinkage toward the diagonal).  NOTE: mlfinlab shrinks with sklearn's OAS estimator
    instead; this case pins the library's own documented rule, not mlfinlab's.
"""

import csv
import json
import pathlib

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

HERE = pathlib.Path(__file__).resolve().parent
PRICES = HERE.parent / "portfolio_optimization" / "stock_prices.csv"


def get_ivp(cov):
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def get_cluster_var(cov, items):
    sub = cov[np.ix_(items, items)]
    w = get_ivp(sub).reshape(-1, 1)
    return float((w.T @ sub @ w)[0, 0])


def get_quasi_diag(link):
    """Snippet 16.2, written recursively: left child first, then right child."""
    n = link.shape[0] + 1

    def walk(node):
        if node < n:
            return [int(node)]
        left, right = int(link[node - n, 0]), int(link[node - n, 1])
        return walk(left) + walk(right)

    return walk(2 * n - 2)


def get_rec_bipart(cov, sort_ix):
    w = np.ones(cov.shape[0])
    clusters = [sort_ix]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left, v_right = get_cluster_var(cov, left), get_cluster_var(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def hrp(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = sch.linkage(squareform(dist, checks=False), method="single")
    order = get_quasi_diag(link)
    weights = get_rec_bipart(cov, order)
    return order, weights


def load_prices():
    with open(PRICES, newline="") as fh:
        rows = list(csv.reader(fh))
    names = rows[0][1:]
    prices = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return names, prices


def cov_from_prices(prices):
    returns = prices[1:] / prices[:-1] - 1.0
    return np.cov(returns, rowvar=False, ddof=1)


def main():
    names, prices = load_prices()
    out = {}

    order, weights = hrp(cov_from_prices(prices))
    out["stock_prices"] = {"names": names, "order": order, "weights": weights.tolist()}

    cov = cov_from_prices(prices)
    shrunk = cov * 0.9
    np.fill_diagonal(shrunk, np.diag(cov))
    order, weights = hrp(shrunk)
    out["stock_prices_shrunk"] = {"order": order, "weights": weights.tolist()}

    weekly = prices[4::5]
    order, weights = hrp(cov_from_prices(weekly))
    out["stock_prices_weekly"] = {"order": order, "weights": weights.tolist()}

    rng = np.random.default_rng(20260919)
    a = rng.normal(size=(40, 8))
    a[:, 1] += 0.8 * a[:, 0]
    a[:, 5] += 0.6 * a[:, 4]
    a *= rng.uniform(0.5, 2.0, size=8)
    cov = np.cov(a, rowvar=False, ddof=1)
    order, weights = hrp(cov)
    out["random_cov_8"] = {"cov": cov.tolist(), "order": order, "weights": weights.tolist()}

    (HERE / "reference.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(k, "order", v["order"], "sum", sum(v["weights"]))


if __name__ == "__main__":
    main()
