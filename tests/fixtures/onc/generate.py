"""Independent reference for crates/openquant/tests/onc_reference.rs.

Run from the repository root:

    uv run --no-project --with numpy --with scikit-learn python tests/fixtures/onc/generate.py

(--no-project keeps numpy/scikit-learn out of this repository's own Python environment.)

ONC (Lopez de Prado & Lewis 2019, snippet 1) scores a clustering of a correlation matrix by
sklearn.metrics.silhouette_samples(X, labels) where X is the distance matrix
D_ij = sqrt((1 - rho_ij) / 2) *used as an observation matrix* (each row of D is a point; the
silhouette metric is Euclidean between rows).

This script builds a sample correlation matrix from simulated returns with three planted
factors (so, unlike an exact block matrix, every silhouette value is different), and records
sklearn's silhouette value of every series under the planted labels. The Rust test asserts that
ONC (a) recovers the planted clusters and (b) reports those silhouette values.
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import silhouette_samples

HERE = pathlib.Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(1987)
    sizes = [4, 6, 5]
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n_obs = 500
    factors = rng.normal(size=(n_obs, len(sizes)))
    noise = rng.normal(size=(n_obs, labels.size))
    loadings = rng.uniform(0.8, 1.2, size=labels.size)
    returns = factors[:, labels] * loadings + 0.6 * noise
    corr = np.corrcoef(returns, rowvar=False)

    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    silh = silhouette_samples(dist, labels)

    out = {
        "corr": corr.tolist(),
        "planted_clusters": [np.flatnonzero(labels == k).tolist() for k in range(len(sizes))],
        "silhouette": silh.tolist(),
    }
    (HERE / "silhouette_reference.json").write_text(json.dumps(out, indent=1) + "\n")
    print("min/max within-block corr", min(corr[i, j] for i in range(15) for j in range(15) if i != j and labels[i] == labels[j]))
    print("max cross-block corr", max(abs(corr[i, j]) for i in range(15) for j in range(15) if labels[i] != labels[j]))
    print("silhouette", np.round(silh, 4))


if __name__ == "__main__":
    main()
