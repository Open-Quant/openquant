"""Independent HCAA reference values for crates/openquant/tests/hcaa_reference.rs and
python/tests/test_core_hcaa.py.

Run from the repository root:

    uv run --no-project --with numpy --with scipy python tests/fixtures/hcaa/generate.py

(--no-project keeps numpy/scipy out of this repository's own Python environment.)

It imports only numpy and scipy (no openquant, no mlfinlab) and shares no code with the Rust
library or with tests/fixtures/hrp/generate.py.  The method is written from its description
(Raffinot 2017, as documented on the hcaa module page): correlation distance
d_ij = sqrt(2 (1 - rho_ij)), a single-linkage tree built by scipy, weight 1 at the root, each
of the top k - 1 merges splitting a node's weight between its two children by the metric
(each side scored as its inverse-variance portfolio), and below the cut each cluster shared by
inverse variance (equal weights for equal_weighting).

Every case is computed with both trees the library offers (openquant.hcaa's `distance` option):

  * "correlation": the condensed matrix d is handed to scipy's linkage(method="single") as
    pairwise distances, as mlfinlab's HCAA does (`linkage(squareform(d))`).
  * "distance_of_distances": the square matrix d is handed to linkage as AFML's Snippet 16.4
    does, which clusters on the Euclidean distance between columns of d,
    d~_ij = sqrt(sum_n (d_ni - d_nj)^2).  The script asserts it equals linkage on pdist(d).

For each distance the case records `link` (the (left, right) merge list: ids below n are assets
and n + k is the cluster formed by merge k), `order` (leaf order, left child first) and
`weights[metric][k]` for metric in minimum_variance / minimum_standard_deviation /
equal_weighting and k in "none" (no cut), "2", "4".

Cases written to reference.json:
  * stock_prices: the 23-ETF price fixture, simple returns, sample covariance (ddof=1).
  * random_cov_8: a seeded random 8x8 covariance passed directly.
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
METRICS = ("minimum_variance", "minimum_standard_deviation", "equal_weighting")
CUTS = (None, 2, 4)


def ivp(cov, items):
    inv = 1.0 / np.diag(cov)[items]
    return inv / inv.sum()


def side_variance(cov, items):
    w = ivp(cov, items)
    return float(w @ cov[np.ix_(items, items)] @ w)


def linkage(dist, distance):
    if distance == "correlation":
        return sch.linkage(squareform(dist, checks=False), method="single")
    # Snippet 16.4 verbatim: the square matrix goes in, so scipy treats rows as observations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sch.ClusterWarning)
        link = sch.linkage(dist, method="single")
    check = sch.linkage(pdist(dist, metric="euclidean"), method="single")
    assert np.array_equal(link[:, [0, 1, 3]], check[:, [0, 1, 3]])
    assert np.allclose(link[:, 2], check[:, 2], rtol=1e-12, atol=0.0)
    return link


def leaves(link, node):
    n = link.shape[0] + 1
    if node < n:
        return [int(node)]
    left, right = int(link[node - n, 0]), int(link[node - n, 1])
    return leaves(link, left) + leaves(link, right)


def alpha(cov, left, right, metric):
    if metric == "equal_weighting":
        return 0.5
    v_left, v_right = side_variance(cov, left), side_variance(cov, right)
    if metric == "minimum_variance":
        return 1.0 - v_left / (v_left + v_right)
    s_left, s_right = np.sqrt(v_left), np.sqrt(v_right)
    return 1.0 - s_left / (s_left + s_right)


def hcaa_weights(cov, link, metric, k):
    n = cov.shape[0]
    k = n if k is None else k
    w = np.zeros(n)
    # scipy's merges are in increasing height, so the top k - 1 merges are the last k - 1 rows.
    split_rows = set(range(n - k, n - 1))

    def walk(node, weight):
        if node < n:
            w[node] = weight
            return
        row = node - n
        if row in split_rows:
            left, right = int(link[row, 0]), int(link[row, 1])
            a = alpha(cov, leaves(link, left), leaves(link, right), metric)
            walk(left, weight * a)
            walk(right, weight * (1.0 - a))
            return
        members = leaves(link, node)
        if metric == "equal_weighting":
            inside = np.full(len(members), 1.0 / len(members))
        else:
            inside = ivp(cov, members)
        w[members] = weight * inside

    walk(2 * n - 2, 1.0)
    return w


def hcaa(cov, distance):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(dist, distance)
    # A tie in merge height would make the tree (and the cut) depend on tie-breaking.
    assert np.all(np.diff(link[:, 2]) > 1e-9)
    pairs = [[int(a), int(b)] for a, b in link[:, :2]]
    assert all(a < b for a, b in pairs)
    weights = {
        m: {("none" if k is None else str(k)): hcaa_weights(cov, link, m, k).tolist() for k in CUTS}
        for m in METRICS
    }
    return {"link": pairs, "order": leaves(link, 2 * cov.shape[0] - 2), "weights": weights}


def both(cov):
    return {d: hcaa(cov, d) for d in ("correlation", "distance_of_distances")}


def load_prices():
    with open(PRICES, newline="") as fh:
        rows = list(csv.reader(fh))
    names = rows[0][1:]
    prices = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return names, prices


def main():
    names, prices = load_prices()
    returns = prices[1:] / prices[:-1] - 1.0
    out = {"stock_prices": {"names": names, **both(np.cov(returns, rowvar=False, ddof=1))}}

    rng = np.random.default_rng(20260925)
    a = rng.normal(size=(40, 8))
    a[:, 1] += 0.8 * a[:, 0]
    a[:, 5] += 0.6 * a[:, 4]
    a[:, 6] += 0.5 * a[:, 1]
    a *= rng.uniform(0.5, 2.0, size=8)
    cov = np.cov(a, rowvar=False, ddof=1)
    out["random_cov_8"] = {"cov": cov.tolist(), **both(cov)}

    (HERE / "reference.json").write_text(json.dumps(out, indent=1) + "\n")
    for key, case in out.items():
        c, dd = case["correlation"], case["distance_of_distances"]
        print(key, "order", c["order"])
        print(
            " " * len(key),
            "d~   ",
            dd["order"],
            "same tree" if dd["link"] == c["link"] else "different tree",
        )


if __name__ == "__main__":
    main()
