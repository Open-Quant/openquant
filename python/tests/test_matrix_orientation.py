"""Matrices cross the Python boundary as rows of observations, columns of assets (#74).

`helpers::matrix_from_rows` once handed row-flattened data to a column-major constructor,
which interleaves observations and assets for any non-square input. Symmetric inputs hid it.
"""

import math
import random

from openquant import _core


def _prices(vols, n_obs=400, seed=11):
    """Independent geometric random walks, one column per asset, with the given daily vols."""
    rng = random.Random(seed)
    rows = [[100.0] * len(vols)]
    for _ in range(n_obs - 1):
        rows.append([p * math.exp(rng.gauss(0.0, v)) for p, v in zip(rows[-1], vols)])
    return rows


def _inverse_variance(prices):
    """Closed form from log returns (the library's convention) and the sample variance."""
    n_assets = len(prices[0])
    inv = []
    for j in range(n_assets):
        rets = [math.log(prices[i][j] / prices[i - 1][j]) for i in range(1, len(prices))]
        mean = sum(rets) / len(rets)
        inv.append(1.0 / (sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)))
    total = sum(inv)
    return [w / total for w in inv]


def test_non_square_prices_keep_their_orientation():
    # 400 observations x 3 assets: far from square, and the assets differ 20x in volatility,
    # so a transposed or interleaved read cannot land on the right weights by accident.
    prices = _prices([0.002, 0.01, 0.04])
    expected = _inverse_variance(prices)

    weights, *_ = _core.portfolio.allocate_inverse_variance(prices)

    assert len(weights) == 3
    assert expected[0] > 0.9  # the calm asset dominates; equal weights would be 1/3
    for got, want in zip(weights, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)
