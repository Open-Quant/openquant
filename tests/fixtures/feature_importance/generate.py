"""Independent reference values for crates/openquant/tests/feature_importance_reference.rs.

Run from the repository root:

    uv run --no-project --with numpy --with scipy python tests/fixtures/feature_importance/generate.py

(--no-project keeps numpy/scipy out of this repository's own Python environment.)

Reproduces the "feature importance vs PCA" check of AFML section 8.4.2 / snippet 8.6 the way
mlfinlab's feature_pca_analysis does, with numpy for the eigen-decomposition and scipy.stats for
the four correlation measures:

    X      standardised features (rows = observations)
    L, W   eigenvalues (descending) and eigenvectors of X'X, truncated at the first component
           where cumulative explained variance reaches `variance_thresh`
    a      = | W[:, c] * L[c] | flattened component by component
    b      = the importance vector tiled once per kept component
    pearson / spearman / kendall = scipy.stats.{pearsonr, spearmanr, kendalltau}(b, a)
    weighted_kendall_rank = scipy.stats.weightedtau(importance, 1 / rank_desc(sum_c |W L|))

Every output is invariant to the ddof used for standardising (it rescales all of `a` by one
constant) and to eigenvector signs (absolute values), so none of this library's conventions
can leak into the expected numbers.

Two cases:
  * one_component: variance_thresh low enough that a single component is kept, so `b` has no
    ties and rank correlations are unambiguous.
  * several_components: `b` is tiled, so it has ties; scipy then uses average ranks for
    Spearman and tau-b for Kendall.
"""

import json
import pathlib

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent


def analyse(x, importance, variance_thresh):
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    evals, evecs = np.linalg.eigh(z.T @ z)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    cum = np.cumsum(evals) / evals.sum()
    kept = int(np.searchsorted(cum, variance_thresh) + 1)
    evals, evecs = evals[:kept], evecs[:, :kept]

    a = np.abs(evecs * evals).T.flatten()  # component by component
    b = np.tile(importance, kept)
    strength = np.abs(evecs * evals).sum(axis=1)
    pca_rank = stats.rankdata(-strength)
    return {
        "variance_thresh": variance_thresh,
        "kept_components": kept,
        "pearson": float(stats.pearsonr(b, a)[0]),
        "spearman": float(stats.spearmanr(b, a)[0]),
        "kendall": float(stats.kendalltau(b, a)[0]),
        "weighted_kendall_rank": float(stats.weightedtau(importance, 1.0 / pca_rank)[0]),
    }


def main():
    rng = np.random.default_rng(8)
    n, m = 40, 5
    common = rng.normal(size=(n, 1))
    x = 0.9 * common + 0.5 * rng.normal(size=(n, m))
    x[:, 3] = rng.normal(size=n)
    x[:, 4] = 0.5 * x[:, 3] + rng.normal(size=n)
    importance = np.array([0.31, 0.24, 0.19, 0.08, 0.18])

    out = {
        "x": x.tolist(),
        "importance": importance.tolist(),
        "one_component": analyse(x, importance, 0.30),
        "several_components": analyse(x, importance, 0.95),
    }
    assert out["one_component"]["kept_components"] == 1
    assert out["several_components"]["kept_components"] > 1
    (HERE / "pca_reference.json").write_text(json.dumps(out, indent=1) + "\n")
    for k in ("one_component", "several_components"):
        print(k, out[k])


if __name__ == "__main__":
    main()
