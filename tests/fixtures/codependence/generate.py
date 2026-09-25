"""Reference values for the codependence tests, from the published definitions in numpy.

    uv run --with numpy --with pandas python tests/fixtures/codependence/generate.py

Independent of this library (and of mlfinlab). Reads random_state_42.csv next to
this file, checks it still regenerates from numpy RandomState(42), and writes
reference.json next to this file.

Sources:
- Angular distances: Lopez de Prado, "Machine Learning for Asset Managers" (MLAM),
  section 3.2: d = sqrt(0.5 (1 - rho)); the absolute and squared variants replace
  rho by |rho| and rho^2.
- Distance correlation: Szekely, Rizzo & Bakirov (2007), "Measuring and testing
  dependence by correlation of distances", Ann. Statist. 35(6), definitions of
  dCov^2_n, dVar^2_n (V-statistics over double-centred distance matrices) and
  dCor^2_n = dCov^2_n(X, Y) / sqrt(dVar^2_n(X) dVar^2_n(Y)).
- Entropies and mutual information: MLAM snippet 3.1 (np.histogram2d contingency
  table, np.histogram marginals, natural log); normalised mutual information
  I(X, Y) / min(H(X), H(Y)) as in the same snippet.
- Variation of information: MLAM snippet 3.2, VI = H(X) + H(Y) - 2 I(X, Y),
  normalised by the joint entropy H(X, Y) = H(X) + H(Y) - I(X, Y).
- Optimal number of bins: MLAM snippet 3.3 (Hacine-Gharbi et al. 2012 univariate,
  Hacine-Gharbi & Ravier 2018 bivariate), with the bivariate count used when no
  bin count is given, as in snippet 3.3's varInfo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "random_state_42.csv", float_precision="round_trip")
x, y_1, y_2 = (df[c].to_numpy(dtype=float) for c in ("x", "y_1", "y_2"))

# The fixture is the seeded draw below. The draw is not bit-identical across platforms
# (libm log/sqrt in the Box-Muller step differ in the last ulp, ~2e-15 here on macOS
# arm64), so the references are computed from the CSV as stored and this only guards
# against the file drifting from its recipe.
state = np.random.RandomState(42)
x_regen = state.normal(size=1000)
y_1_regen = x_regen ** 2 + state.normal(size=1000) / 5
y_2_regen = abs(x_regen) + state.normal(size=1000) / 5
regen_max_abs_diff = float(max(np.max(np.abs(a - b)) for a, b in
                               ((x, x_regen), (y_1, y_1_regen), (y_2, y_2_regen))))
assert regen_max_abs_diff < 1e-12, regen_max_abs_diff


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def angular(a, b, kind):
    rho = corr(a, b)
    rho = {"plain": rho, "absolute": abs(rho), "squared": rho ** 2}[kind]
    return float(np.sqrt(0.5 * (1.0 - rho)))


def distance_correlation(a, b):
    def centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(axis=0)[None, :] - d.mean(axis=1)[:, None] + d.mean()

    A, B = centred(a), centred(b)
    dcov2_xy = (A * B).mean()
    dvar2_x = (A * A).mean()
    dvar2_y = (B * B).mean()
    return float(np.sqrt(dcov2_xy / np.sqrt(dvar2_x * dvar2_y)))


def num_bins(n_obs, corr=None):
    # MLAM snippet 3.3.
    if corr is None:
        z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs ** 2) ** 0.5) ** (1 / 3.0)
        b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    else:
        b = round(2 ** -0.5 * (1 + (1 + 24 * n_obs / (1.0 - corr ** 2)) ** 0.5) ** 0.5)
    return int(b)


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def mutual_info_from_contingency(c):
    # I(X, Y) = sum_ij p_ij ln(p_ij / (p_i p_j)) over the non-empty cells.
    p = c / c.sum()
    p_i = p.sum(axis=1, keepdims=True)
    p_j = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (p_i * p_j)[nz])).sum())


def info_metrics(a, b, bins):
    # MLAM snippets 3.1 and 3.2.
    c_xy = np.histogram2d(a, b, bins)[0]
    h_x = entropy(np.histogram(a, bins)[0])
    h_y = entropy(np.histogram(b, bins)[0])
    i_xy = mutual_info_from_contingency(c_xy)
    v_xy = h_x + h_y - 2 * i_xy
    return {
        "mutual_info": i_xy,
        "mutual_info_normalised": i_xy / min(h_x, h_y),
        "variation_of_information": v_xy,
        "variation_of_information_normalised": v_xy / (h_x + h_y - i_xy),
    }


n = len(x)
rho_x_y1 = corr(x, y_1)
bins_univariate = num_bins(n)
bins_bivariate = num_bins(n, rho_x_y1)
optimal = info_metrics(x, y_1, bins_bivariate)
ten = info_metrics(x, y_1, 10)

out = {
    "source": "MLAM ch. 3 snippets 3.1-3.3 and Szekely, Rizzo & Bakirov (2007), in numpy %s "
              "on tests/fixtures/codependence/random_state_42.csv" % np.__version__,
    "n_obs": n,
    "corr_x_y1": rho_x_y1,
    "angular_distance_x_y1": angular(x, y_1, "plain"),
    "absolute_angular_distance_x_y1": angular(x, y_1, "absolute"),
    "squared_angular_distance_x_y1": angular(x, y_1, "squared"),
    "distance_correlation_x_y1": distance_correlation(x, y_1),
    "distance_correlation_x_y2": distance_correlation(x, y_2),
    "optimal_bins_univariate": bins_univariate,
    "optimal_bins_bivariate_x_y1": bins_bivariate,
    "mutual_info_x_y1": optimal["mutual_info"],
    "mutual_info_normalised_x_y1": optimal["mutual_info_normalised"],
    "mutual_info_x_y1_10_bins": ten["mutual_info"],
    "variation_of_information_x_y1": optimal["variation_of_information"],
    "variation_of_information_normalised_x_y1": optimal["variation_of_information_normalised"],
    "variation_of_information_x_y1_10_bins": ten["variation_of_information"],
}
(HERE / "reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
