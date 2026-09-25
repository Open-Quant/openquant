"""ONC clusters of the breast-cancer feature correlation matrix, independent of this library.

    uv run --no-project --with numpy --with pandas --with scikit-learn python tests/fixtures/onc/generate_breast_cancer.py

Writes breast_cancer_reference.json next to this file. Imports neither openquant nor mlfinlab.

ONC as published: Lopez de Prado & Lewis (2019), "Detection of false investment strategies using
unsupervised learning methods", and "Machine Learning for Asset Managers" snippets 4.1
(clusterKMeansBase) and 4.2 (makeNewOutputs, clusterKMeansTop), with scikit-learn's KMeans and
silhouette_samples. The observation matrix is the correlation distance sqrt((1 - rho) / 2),
each row a point; for each k in 2..n-1 KMeans runs `repeat` times with a fresh random start
(seeded here, so the script is reproducible), and the clustering with the highest
mean(silhouette) / std(silhouette) is kept. Clusters whose t-statistic is below the average
are re-clustered recursively, and the result is kept only if it improves.

Input: tests/fixtures/onc/breast_cancer.csv (scikit-learn's copy of the UCI data; the first line
is scikit-learn's header, the last column the target), 30 features, sample correlation.

The cluster sets are recorded. KMeans depends on its random starts, so the script runs ONC under
several seeds and records only the clusters that every run finds ("stable_clusters"): those are a
property of the data, not of a seed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

HERE = Path(__file__).parent
raw = pd.read_csv(HERE / "breast_cancer.csv", header=None, skiprows=1)
features = raw.iloc[:, :30].astype(float)
corr0 = features.corr()  # pandas sample correlation


def cluster_kmeans_base(corr0, max_num_clusters, n_init, rng):  # snippet 4.1
    x = ((1 - corr0.fillna(0)) / 2.0) ** 0.5
    silh, kmeans, best = pd.Series(dtype=float), None, -np.inf
    for _ in range(n_init):
        for i in range(2, max_num_clusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1, random_state=int(rng.integers(2**31)))
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = silh_.mean() / silh_.std()
            if kmeans is None or np.isnan(best) or stat > best:
                silh, kmeans, best = silh_, kmeans_, stat
    new_idx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[new_idx].iloc[:, new_idx]
    clstrs = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=x.index)
    return corr1, clstrs, silh


def make_new_outputs(corr0, clstrs, clstrs2):  # snippet 4.2
    clstrs_new = {}
    for i in clstrs:
        clstrs_new[len(clstrs_new)] = list(clstrs[i])
    for i in clstrs2:
        clstrs_new[len(clstrs_new)] = list(clstrs2[i])
    new_idx = [j for i in clstrs_new for j in clstrs_new[i]]
    corr_new = corr0.loc[new_idx, new_idx]
    x = ((1 - corr0.fillna(0)) / 2.0) ** 0.5
    kmeans_labels = np.zeros(len(x.columns))
    for i in clstrs_new:
        idxs = [x.index.get_loc(k) for k in clstrs_new[i]]
        kmeans_labels[idxs] = i
    silh_new = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    return corr_new, clstrs_new, silh_new


def cluster_kmeans_top(corr0, max_num_clusters, n_init, rng):  # snippet 4.2
    corr1, clstrs, silh = cluster_kmeans_base(corr0, min(max_num_clusters, corr0.shape[1] - 1), n_init, rng)
    cluster_tstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs}
    tstat_mean = sum(cluster_tstats.values()) / len(cluster_tstats)
    redo = [i for i in cluster_tstats if cluster_tstats[i] < tstat_mean]
    if len(redo) <= 1:
        return corr1, clstrs, silh
    keys_redo = [j for i in redo for j in clstrs[i]]
    corr_tmp = corr0.loc[keys_redo, keys_redo]
    _, clstrs2, _ = cluster_kmeans_top(corr_tmp, min(max_num_clusters, corr_tmp.shape[1] - 1), n_init, rng)
    corr_new, clstrs_new, silh_new = make_new_outputs(
        corr0, {i: clstrs[i] for i in clstrs if i not in redo}, clstrs2)
    new_tstat_mean = np.mean([np.mean(silh_new[clstrs_new[i]]) / np.std(silh_new[clstrs_new[i]]) for i in clstrs_new])
    if new_tstat_mean <= tstat_mean:
        return corr1, clstrs, silh
    return corr_new, clstrs_new, silh_new


runs = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    _, clstrs, _ = cluster_kmeans_top(corr0, corr0.shape[1] - 1, 10, rng)
    runs.append(sorted(sorted(int(j) for j in members) for members in clstrs.values()))
    print("seed", seed, len(runs[-1]), "clusters:", runs[-1])

stable = [c for c in runs[0] if all(c in r for r in runs[1:])]
out = {
    "source": "tests/fixtures/onc/generate_breast_cancer.py: MLAM snippets 4.1-4.2 with scikit-learn "
              "KMeans/silhouette_samples, 5 seeds x 10 restarts",
    "corr_0_2": float(corr0.iloc[0, 2]),
    "runs": runs,
    "stable_clusters": stable,
}
(HERE / "breast_cancer_reference.json").write_text(json.dumps(out, indent=2) + "\n")
print("stable:", stable)
