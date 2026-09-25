"""Independent HRP reference values for crates/openquant/tests/hrp_reference.rs.

Run from the repository root:

    uv run --no-project --with numpy --with scipy python tests/fixtures/hrp/generate.py

(--no-project keeps numpy/scipy out of this repository's own Python environment.)

This is a direct transcription of Lopez de Prado, "Advances in Financial Machine Learning",
snippets 16.1-16.4 (getIVP, getClusterVar, getQuasiDiag, getRecBipart), with the clustering
step done by scipy.  It shares no code with the Rust library.  Every case is computed with both
trees the library offers (openquant.hrp's `distance` option):

  * "correlation" (the top-level `link`, `order`, `weights` of each case): the condensed
    correlation-distance matrix d_ij = sqrt((1 - rho_ij) / 2) is handed to
    scipy.cluster.hierarchy.linkage(method="single") as pairwise distances, as mlfinlab does.
  * "distance_of_distances" (the nested object of that name): the square matrix d is handed to
    linkage exactly as the book's snippet 16.4 does (`sch.linkage(dist, 'single')`).  scipy
    reads a square array as one observation per row, so this clusters on the Euclidean
    distance between columns of d, d~_ij = sqrt(sum_n (d_ni - d_nj)^2), the second step of
    AFML section 16.4.1.  The script asserts that it equals linkage on condensed pdist(d).

`link` is the (left, right) merge list of scipy's linkage matrix: ids below n are assets and
n + k is the cluster formed by merge k.

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
import warnings

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

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


def linkage(dist, distance):
    if distance == "correlation":
        return sch.linkage(squareform(dist, checks=False), method="single")
    # Snippet 16.4 verbatim: the square matrix goes in, so scipy treats rows as observations
    # (and warns that it "looks suspiciously like an uncondensed distance matrix").
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sch.ClusterWarning)
        link = sch.linkage(dist, method="single")
    check = sch.linkage(pdist(dist, metric="euclidean"), method="single")
    assert np.array_equal(link[:, [0, 1, 3]], check[:, [0, 1, 3]])
    assert np.allclose(link[:, 2], check[:, 2], rtol=1e-12, atol=0.0)
    return link


def hrp(cov, distance):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(dist, distance)
    pairs = [[int(a), int(b)] for a, b in link[:, :2]]
    assert all(a < b for a, b in pairs)
    order = get_quasi_diag(link)
    weights = get_rec_bipart(cov, order)
    return {"link": pairs, "order": order, "weights": weights.tolist()}


def both(cov):
    out = hrp(cov, "correlation")
    out["distance_of_distances"] = hrp(cov, "distance_of_distances")
    return out


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

    out["stock_prices"] = {"names": names, **both(cov_from_prices(prices))}

    cov = cov_from_prices(prices)
    shrunk = cov * 0.9
    np.fill_diagonal(shrunk, np.diag(cov))
    out["stock_prices_shrunk"] = both(shrunk)

    weekly = prices[4::5]
    out["stock_prices_weekly"] = both(cov_from_prices(weekly))

    rng = np.random.default_rng(20260919)
    a = rng.normal(size=(40, 8))
    a[:, 1] += 0.8 * a[:, 0]
    a[:, 5] += 0.6 * a[:, 4]
    a *= rng.uniform(0.5, 2.0, size=8)
    cov = np.cov(a, rowvar=False, ddof=1)
    out["random_cov_8"] = {"cov": cov.tolist(), **both(cov)}

    (HERE / "reference.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        dd = v["distance_of_distances"]
        print(k, "order", v["order"], "sum", sum(v["weights"]))
        print(" " * len(k), "d~   ", dd["order"], "same tree" if dd["link"] == v["link"] else "")


if __name__ == "__main__":
    main()
