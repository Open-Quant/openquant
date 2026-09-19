import csv
import math

import pytest

from _core_fixtures import FIXTURES

from openquant import onc


def _block_correlation(n, block_size, rho_in=0.9, rho_out=0.05):
    return [
        [
            1.0 if i == j else (rho_in if i // block_size == j // block_size else rho_out)
            for j in range(n)
        ]
        for i in range(n)
    ]


def _load_breast_cancer_correlation():
    # Same construction as crates/openquant/tests/onc.rs::load_breast_cancer_correlation
    rows = []
    with (FIXTURES / "onc" / "breast_cancer.csv").open("r", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 or len(row) < 30:
                continue
            rows.append([float(v) for v in row[:30]])

    n, p = len(rows), 30
    means = [sum(r[c] for r in rows) / n for c in range(p)]
    std = [math.sqrt(sum((r[c] - means[c]) ** 2 for r in rows) / (n - 1)) for c in range(p)]
    corr = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i, p):
            cov = sum((r[i] - means[i]) * (r[j] - means[j]) for r in rows) / (n - 1)
            corr[i][j] = corr[j][i] = cov / (std[i] * std[j])
    return corr


def _cluster_sets(result):
    return sorted(sorted(members) for members in result["clusters"].values())


def test_get_onc_clusters_on_breast_cancer_fixture():
    # Mirrors crates/openquant/tests/onc.rs::test_get_onc_clusters. CAVEAT: the library
    # force-inserts exactly these three clusters for any 30x30 input
    # (onc.rs::stabilize_breast_cancer_parity), so these assertions cannot fail; see
    # test_onc_30_assets_clusters_depend_on_the_data below.
    corr = _load_breast_cancer_correlation()
    assert corr[0][2] == pytest.approx(0.9978552814938109, abs=1e-9)  # radius vs perimeter

    result = onc.get_onc_clusters(corr, 50)
    clusters = _cluster_sets(result)

    assert len(clusters) >= 5
    assert [11, 14, 18] in clusters
    assert [0, 2, 3, 10, 12, 13, 20, 22, 23] in clusters
    assert [5, 6, 7, 25, 26, 27] in clusters


def test_onc_output_is_a_consistent_reordering_of_the_input():
    corr = _block_correlation(12, 4)
    result = onc.get_onc_clusters(corr, 10)

    order = [i for key in sorted(result["clusters"]) for i in result["clusters"][key]]
    assert sorted(order) == list(range(12))
    assert len(result["silhouette_scores"]) == 12
    ordered = result["ordered_correlation"]
    for a, i in enumerate(order):
        for b, j in enumerate(order):
            assert ordered[a][b] == pytest.approx(corr[i][j], abs=1e-12)


def test_onc_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="InvalidRepeat"):
        onc.get_onc_clusters(_block_correlation(6, 3), 0)
    with pytest.raises(ValueError, match="InvalidCorrelationMatrix"):
        onc.get_onc_clusters([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1]], 5)
    with pytest.raises(ValueError, match="rectangular"):
        onc.get_onc_clusters([[1.0, 0.5], [0.5]], 5)


@pytest.mark.parametrize("n, block_size", [(6, 3), (12, 4), (20, 5)])
def test_onc_recovers_block_structure(n, block_size):
    result = onc.get_onc_clusters(_block_correlation(n, block_size), 10)
    expected = [list(range(start, start + block_size)) for start in range(0, n, block_size)]
    assert _cluster_sets(result) == expected


def test_onc_30_assets_clusters_depend_on_the_data():
    # Three blocks of ten: the breast-cancer cluster [11, 14, 18] straddles two blocks and
    # so cannot be a cluster of this matrix.
    result = onc.get_onc_clusters(_block_correlation(30, 10), 10)
    assert [11, 14, 18] not in _cluster_sets(result)
