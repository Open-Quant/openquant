"""openquant.backtesting_engine: CPCV paths from precomputed out-of-sample returns.

Fixtures and expected values are those of crates/openquant/tests/backtesting_engine.rs and
backtesting_engine_reference.rs. The Rust tests pass an evaluator closure; here the same
values are computed up front from `cross_validation.cpcv_splits` and passed in.
"""

from math import comb, sqrt

import numpy as np
import pytest
from openquant import backtesting_engine as bt
from openquant import cross_validation as cv

SAFEGUARDS = {
    "survivorship_bias_control": "n/a (synthetic)",
    "look_ahead_control": "n/a (synthetic)",
    "data_mining_control": "n/a (synthetic)",
    "cost_assumption": "n/a (synthetic)",
    "multiple_testing_control": "n/a (synthetic)",
}
RUN = {"mode_provenance": "reference-tests", "trials_count": 1, "safeguards": SAFEGUARDS}


def point_labels(n):
    """Rust `point_labels`: each label starts and ends on its own day; nothing overlaps."""
    t = np.datetime64("2024-01-01", "ns") + np.arange(n) * np.timedelta64(1, "D")
    return t, t


def minute_labels(n):
    """Rust `build_data` spans: one label a minute from 09:30, each lasting 3 minutes."""
    t0 = np.datetime64("2024-01-01T09:30", "ns") + np.arange(n) * np.timedelta64(1, "m")
    return t0, t0 + np.timedelta64(3, "m")


def test_cpcv_path_count_matches_phi_formula():
    # Rust: cpcv_path_count_matches_phi_formula
    assert bt.cpcv_path_count(6, 2) == 5
    assert bt.cpcv_path_count(8, 3) == 21
    with pytest.raises(ValueError):
        bt.cpcv_path_count(4, 4)


def test_cpcv_split_and_path_counts_match_combinatorics():
    # Rust: cpcv_split_and_path_counts_match_combinatorics
    for n_groups in range(2, 10):
        for k in range(1, n_groups):
            t0, t1 = point_labels(2 * n_groups)
            splits = cv.cpcv_splits(t0, t1, n_groups, k, 0.0)
            echo = [s["test_indices"].astype(float) for s in splits]
            res = bt.run_cpcv(
                t0, t1, echo, n_groups=n_groups, test_groups=k, pct_embargo=0.0, **RUN
            )
            assert bt.cpcv_path_count(n_groups, k) == comb(n_groups - 1, k - 1)
            assert len(res["splits"]) == comb(n_groups, k)
            assert len(res["path_distribution"]) == comb(n_groups - 1, k - 1)


def test_cpcv_embargo_follows_each_test_block():
    # Rust: cpcv_embargo_follows_each_test_block (backtesting_engine_reference.rs). 12 point
    # labels in 6 groups of 2, k = 2, h = ceil(0.05 * 12) = 1; the embargo only follows a block.
    #   split 0, groups (0, 1): test 0..4, one block; embargo 4. train = 5..12.
    #   split 1, groups (0, 2): test {0, 1, 4, 5}; embargo 2 and 6. train = {3} + 7..12.
    t0, t1 = point_labels(12)
    splits = cv.cpcv_splits(t0, t1, 6, 2, 0.05)
    echo = [s["test_indices"].astype(float) for s in splits]
    res = bt.run_cpcv(t0, t1, echo, n_groups=6, test_groups=2, pct_embargo=0.05, **RUN)
    first, second = res["splits"][0], res["splits"][1]
    assert first["train_indices"] == list(range(5, 12))
    assert first["embargo_count"] == 1
    assert second["test_indices"] == [0, 1, 4, 5]
    assert second["train_indices"] == [3, *range(7, 12)]
    assert second["embargo_count"] == 2


def test_cpcv_paths_follow_afml_assignment_and_cover_every_sample_once():
    # Rust: cpcv_paths_follow_afml_assignment_and_cover_every_sample_once. The evaluator's
    # `1000 * split_id + index` is computed here per split.
    n_groups, per_group = 6, 2
    n = n_groups * per_group
    t0, t1 = point_labels(n)
    splits = cv.cpcv_splits(t0, t1, n_groups, 2, 0.0)
    returns = [(1000 * s["split_id"] + s["test_indices"]).astype(float) for s in splits]
    res = bt.run_cpcv(t0, t1, returns, n_groups=n_groups, test_groups=2, pct_embargo=0.0, **RUN)

    pairs = [(i, j) for i in range(n_groups) for j in range(i + 1, n_groups)]
    for split_id, (i, j) in enumerate(pairs):
        want_test = [x for x in range(n) if x // per_group in (i, j)]
        s = res["splits"][split_id]
        assert s["test_indices"] == want_test
        assert s["train_indices"] == [x for x in range(n) if x not in want_test]
        assert s["test_groups"] == [i, j]

    assert res["path_count"] == 5
    grand_total = 0.0
    for p, path in enumerate(res["path_distribution"]):
        assert path["observations"] == n
        want_sum = 0.0
        for g in range(n_groups):
            with_g = [s for s, pair in enumerate(pairs) if g in pair]
            want_sum += sum(
                1000 * with_g[p] + idx for idx in range(g * per_group, (g + 1) * per_group)
            )
        assert path["mean_return"] == pytest.approx(want_sum / n, abs=1e-9)
        grand_total += path["mean_return"] * n
    assert grand_total == pytest.approx(sum(r.sum() for r in returns), abs=1e-6)

    # The engine numbers paths as cross_validation.cpcv_paths does.
    assert res["path_assignments"] == cv.cpcv_paths(n_groups, 2).tolist()


def test_cpcv_enforces_purge_embargo_and_returns_path_distribution():
    # Rust: cpcv_enforces_purge_embargo_and_returns_path_distribution (backtesting_engine.rs)
    n = 30
    t0, t1 = minute_labels(n)
    data_returns = np.array(
        [(-1.0 if i % 3 == 0 else 1.0) * (0.001 + i * 0.0002) for i in range(n)]
    )
    splits = cv.cpcv_splits(t0, t1, 6, 2, 0.1)
    returns = [
        data_returns[s["test_indices"]]
        + s["split_id"] * 2e-5
        + np.arange(len(s["test_indices"])) * 1e-6
        for s in splits
    ]
    res = bt.run_cpcv(
        t0,
        t1,
        returns,
        n_groups=6,
        test_groups=2,
        pct_embargo=0.1,
        mode_provenance="afml_ch11_ch12_CombinatorialPurgedCrossValidation",
        trials_count=19,
        safeguards=SAFEGUARDS,
    )
    assert res["path_count"] == bt.cpcv_path_count(6, 2)
    assert len(res["path_assignments"]) == len(res["path_distribution"]) == 5
    assert len(res["folds"]) == len(res["splits"]) == 15
    diag = res["diagnostics"]
    assert diag["mode"] == "combinatorial_purged_cross_validation"
    assert diag["split_count"] == 15
    assert diag["trials_count"] == 19
    assert diag["safeguards"] == SAFEGUARDS
    assert diag["total_purged"] > 0 and diag["total_embargoed"] > 0
    assert len({round(p["sharpe"], 12) for p in res["path_distribution"]}) >= 2
    for s in res["splits"]:
        tr, te = np.array(s["train_indices"]), np.array(s["test_indices"])
        assert cv.count_train_test_overlaps(t0, t1, tr, te) == 0


def test_engine_tests_the_same_samples_as_cross_validation():
    # run_cpcv reads caller returns aligned to cross_validation's test indices; both modules
    # must test the same samples in the same split order, whatever the labels and embargo.
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(20, 60))
        n_groups = int(rng.integers(3, 7))
        k = int(rng.integers(1, n_groups))
        starts = np.cumsum(rng.integers(1, 4, size=n))
        t0, t1 = starts, starts + rng.integers(0, 5, size=n)
        splits = cv.cpcv_splits(t0, t1, n_groups, k, 0.0)
        zeros = [np.zeros(len(s["test_indices"])) for s in splits]
        res = bt.run_cpcv(t0, t1, zeros, n_groups=n_groups, test_groups=k, pct_embargo=0.0, **RUN)
        for mine, theirs in zip(splits, res["splits"]):
            assert mine["test_indices"].tolist() == theirs["test_indices"]
            assert list(mine["test_fold_ids"]) == theirs["test_groups"]


def test_assemble_cpcv_paths_matches_the_engine_path_statistics():
    n = 30
    t0, t1 = minute_labels(n)
    splits = cv.cpcv_splits(t0, t1, 6, 2, 0.1)
    paths = cv.cpcv_paths(6, 2)
    rng = np.random.default_rng(11)
    returns = [rng.normal(0.001, 0.01, size=len(s["test_indices"])) for s in splits]

    stitched = bt.assemble_cpcv_paths(returns, splits, paths)
    assert stitched.shape == (5, n)
    assert not np.isnan(stitched).any()

    res = bt.run_cpcv(t0, t1, returns, n_groups=6, test_groups=2, pct_embargo=0.1, **RUN)
    for row, path in zip(stitched, res["path_distribution"]):
        assert path["observations"] == n
        assert row.mean() == pytest.approx(path["mean_return"], rel=1e-12)
        assert row.std(ddof=1) == pytest.approx(path["std_return"], rel=1e-12)
        assert row.mean() / row.std(ddof=1) * sqrt(n) == pytest.approx(path["sharpe"], rel=1e-10)

    # Every (fold, split testing it) value is used by exactly one path.
    echo = [(1000 * s["split_id"] + s["test_indices"]).astype(float) for s in splits]
    used = bt.assemble_cpcv_paths(echo, splits, paths)
    assert sorted(used.ravel().tolist()) == sorted(np.concatenate(echo).tolist())


def test_run_cpcv_rejects_bad_input():
    t0, t1 = point_labels(12)
    splits = cv.cpcv_splits(t0, t1, 6, 2, 0.0)
    good = [np.ones(len(s["test_indices"])) for s in splits]
    kw = {"n_groups": 6, "test_groups": 2, "pct_embargo": 0.0, **RUN}

    with pytest.raises(ValueError, match="split_returns has 14 entries"):
        bt.run_cpcv(t0, t1, good[:-1], **kw)
    with pytest.raises(ValueError, match="15 splits"):
        bt.run_cpcv(t0, t1, good + [np.ones(4)], **kw)
    with pytest.raises(ValueError, match=r"split_returns\[3\] has 3 values"):
        bt.run_cpcv(t0, t1, good[:3] + [np.ones(3)] + good[4:], **kw)
    with pytest.raises(ValueError, match="non-finite"):
        bt.run_cpcv(t0, t1, [np.full(len(g), np.nan) for g in good], **kw)
    missing = {k: v for k, v in SAFEGUARDS.items() if k != "cost_assumption"}
    with pytest.raises(ValueError, match="missing 'cost_assumption'"):
        bt.run_cpcv(t0, t1, good, **{**kw, "safeguards": missing})
    with pytest.raises(ValueError, match="cost_assumption cannot be empty"):
        bt.run_cpcv(t0, t1, good, **{**kw, "safeguards": {**SAFEGUARDS, "cost_assumption": " "}})
    with pytest.raises(ValueError, match="mode_provenance cannot be empty"):
        bt.run_cpcv(t0, t1, good, **{**kw, "mode_provenance": ""})
    with pytest.raises(ValueError, match="t0 is required"):
        bt.run_cpcv(None, t1, good, **kw)
    with pytest.raises(TypeError):
        bt.run_cpcv(t0, t1, good, n_groups=6, test_groups=2, pct_embargo=0.0)
