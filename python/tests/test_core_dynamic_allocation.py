"""AFML Chapter 21 dynamic allocation (issue #112), checked against the book's snippets."""

import itertools
import math
import random

import pytest

from openquant import dynamic_allocation as da


# --- AFML Snippets 21.1-21.3, transcribed in pure Python (numpy replaced by lists) ---------


def book_pigeon_hole(k, n):
    for j in itertools.combinations_with_replacement(range(n), k):
        r = [0] * n
        for i in j:
            r[i] += 1
        yield r


def book_all_weights(k, n):
    out = []
    for part in book_pigeon_hole(k, n):
        for signs in itertools.product([-1, 1], repeat=n):
            out.append([s * p / k for s, p in zip(signs, part)])
    return out


def book_eval(w, means, covs, costs, initial):
    prev, mean, var = initial, 0.0, 0.0
    for wh, mu, cov, c in zip(w, means, covs, costs):
        tcost = sum(ci * abs(a - b) ** 0.5 for ci, a, b in zip(c, wh, prev))
        mean += sum(a * m for a, m in zip(wh, mu)) - tcost
        var += sum(wh[i] * cov[i][j] * wh[j] for i in range(len(wh)) for j in range(len(wh)))
        prev = wh
    return mean / var**0.5


def book_dyn_opt_port(means, covs, costs, k, initial):
    omega = book_all_weights(k, len(means[0]))
    best, best_w = None, None
    for traj in itertools.product(omega, repeat=len(means)):
        sr = book_eval(list(traj), means, covs, costs, initial)
        if best is None or best < sr:
            best, best_w = sr, [list(w) for w in traj]
    return best_w, best


def random_problem(rng, n, h):
    means, covs, costs = [], [], []
    for _ in range(h):
        a = [[rng.uniform(-0.2, 0.2) for _ in range(n)] for _ in range(n)]
        cov = [
            [sum(a[i][m] * a[j][m] for m in range(n)) + (0.01 if i == j else 0.0) for j in range(n)]
            for i in range(n)
        ]
        means.append([rng.uniform(-0.05, 0.05) for _ in range(n)])
        covs.append(cov)
        costs.append([rng.uniform(0.0, 0.01) for _ in range(n)])
    return means, covs, costs


# --- tests ---------------------------------------------------------------------------------


@pytest.mark.parametrize("k,n", [(0, 3), (2, 3), (4, 2), (3, 4)])
def test_partitions_match_snippet_21_1(k, n):
    parts = da.pigeonhole_partitions(k, n)
    assert parts == list(book_pigeon_hole(k, n))
    assert len(parts) == math.comb(k + n - 1, n - 1)


@pytest.mark.parametrize("k,n", [(1, 2), (2, 3), (3, 3), (4, 2)])
def test_omega_is_snippet_21_2_without_repeats(k, n):
    omega = da.all_weights(k, n)
    first_seen = []
    for w in book_all_weights(k, n):
        if w not in first_seen:  # -0.0 == 0.0
            first_seen.append(w)
    assert omega == first_seen
    assert all(math.isclose(sum(abs(x) for x in w), 1.0) for w in omega)


def test_hand_worked_case():
    means = [[0.02, 0.01], [-0.01, 0.03]]
    covs = [[[0.04, 0.0], [0.0, 0.01]]] * 2
    costs = [[0.001, 0.001]] * 2

    switch = [[1.0, 0.0], [0.0, 1.0]]
    assert da.transaction_costs(switch, means, covs, costs) == pytest.approx([0.001, 0.002])
    assert da.trajectory_sharpe_ratio(switch, means, covs, costs) == pytest.approx(
        0.047 / math.sqrt(0.05)
    )

    best = da.dynamic_optimal_portfolio(means, covs, costs, k=1)
    assert best["weights"] == [[0.0, 1.0], [0.0, 1.0]]
    assert best["sharpe_ratio"] == pytest.approx(0.039 / math.sqrt(0.02))
    assert best["transaction_costs"] == pytest.approx([0.001, 0.0])
    assert best["trajectories_evaluated"] == 16


@pytest.mark.parametrize("n,k,h", [(2, 2, 2), (3, 2, 2), (2, 1, 3)])
def test_search_matches_snippet_21_3(n, k, h):
    rng = random.Random(100 * n + 10 * k + h)
    means, covs, costs = random_problem(rng, n, h)
    initial = [0.0] * n
    want_w, want_sr = book_dyn_opt_port(means, covs, costs, k, initial)
    got = da.dynamic_optimal_portfolio(means, covs, costs, k=k)
    assert got["sharpe_ratio"] == pytest.approx(want_sr, abs=1e-12)
    assert got["weights"] == want_w


def test_k_defaults_to_the_number_of_assets():
    rng = random.Random(3)
    means, covs, costs = random_problem(rng, 2, 2)
    default = da.dynamic_optimal_portfolio(means, covs, costs)
    explicit = da.dynamic_optimal_portfolio(means, covs, costs, k=2)
    assert default == explicit


def test_trajectory_cap_raises():
    rng = random.Random(5)
    means, covs, costs = random_problem(rng, 3, 5)
    with pytest.raises(ValueError, match="more than max_trajectories = 1000000"):
        da.dynamic_optimal_portfolio(means, covs, costs)
    with pytest.raises(ValueError, match="more than max_trajectories = 10"):
        da.dynamic_optimal_portfolio(means[:1], covs[:1], costs[:1], max_trajectories=10)


def test_invalid_inputs_raise():
    means = [[0.02, 0.01]]
    costs = [[0.001, 0.001]]
    with pytest.raises(ValueError, match="symmetric positive definite"):
        da.dynamic_optimal_portfolio(means, [[[1.0, 1.0], [1.0, 1.0]]], costs)
    with pytest.raises(ValueError, match="must be square"):
        da.dynamic_optimal_portfolio(means, [[[1.0, 0.0], [0.0]]], costs)
    with pytest.raises(ValueError, match="one entry per horizon"):
        da.dynamic_optimal_portfolio(means, [], costs)
    with pytest.raises(ValueError, match="non-negative"):
        da.dynamic_optimal_portfolio(means, [[[1.0, 0.0], [0.0, 1.0]]], [[-0.1, 0.0]])
