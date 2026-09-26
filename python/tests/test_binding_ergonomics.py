"""Call-site ergonomics of the bindings (issue #77): return types, defaults, keywords."""

import random
import warnings

import pytest
from openquant import ensemble, labeling, sampling, sb_bagging, synthetic_bt


def _is_int_list(values):
    return type(values) is list and all(type(v) is int for v in values)


def test_u8_outputs_are_lists_of_ints_not_bytes():
    vote = ensemble.aggregate_classification_vote([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    assert _is_int_list(vote) and vote == [1, 1, 1]

    _, labels = ensemble.aggregate_classification_probability_mean([[0.9, 0.2], [0.7, 0.4]], 0.5)
    assert _is_int_list(labels) and labels == [1, 0]

    ind_mat = sampling.get_ind_matrix([(0, 2), (1, 3)], [0, 1, 2, 3])
    assert type(ind_mat) is list and all(_is_int_list(row) for row in ind_mat)
    assert ind_mat == [[1, 0], [1, 1], [1, 1], [0, 1]]
    # The list-of-lists output is accepted back as input, as the old list of bytes was.
    assert sampling.get_ind_mat_average_uniqueness(ind_mat) == pytest.approx(
        sampling.get_ind_mat_average_uniqueness([bytes(row) for row in ind_mat])
    )

    n = 20
    x = [[float(i % 5)] for i in range(n)]
    y = [int(row[0] >= 2) for row in x]
    ind = sampling.get_ind_matrix([(i, i + 2) for i in range(n)], list(range(n + 3)))
    fit = sb_bagging.fit_predict_sb_classifier(x, y, ind, n_estimators=5, random_state=3)
    assert _is_int_list(fit["predictions"]) and len(fit["predictions"]) == n
    assert set(fit["predictions"]) <= {0, 1}


STAMPS = [f"2024-01-0{d} 00:00:00" for d in range(1, 8)]
CLOSE = [100.0 + d for d in range(7)]


def test_add_vertical_barrier_defaults_to_zero_and_takes_keywords():
    events = STAMPS[:3]
    two_days = [(STAMPS[i], STAMPS[i + 2]) for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert labeling.add_vertical_barrier(events, STAMPS, num_days=2) == two_days
        assert labeling.add_vertical_barrier(events, STAMPS, None, 2, 0, 0, 0) == two_days
        assert labeling.add_vertical_barrier(events, STAMPS, num_hours=48) == two_days
        # Components add up: one day plus 24 hours is two days.
        assert labeling.add_vertical_barrier(events, STAMPS, num_days=1, num_hours=24) == two_days
        assert labeling.add_vertical_barrier(events, STAMPS, num_minutes=1) == [
            (STAMPS[i], STAMPS[i + 1]) for i in range(3)
        ]


def test_add_vertical_barrier_close_prices_is_deprecated():
    # Issue #194: close_prices was silently ignored (a vertical barrier depends only on the
    # bar times). Passing it still works, positionally or by keyword, but warns.
    events = STAMPS[:3]
    expected = labeling.add_vertical_barrier(events, STAMPS, num_days=2)
    with pytest.warns(DeprecationWarning, match="close_prices is deprecated"):
        assert labeling.add_vertical_barrier(events, STAMPS, CLOSE, 2, 0, 0, 0) == expected
    with pytest.warns(DeprecationWarning, match="close_prices is deprecated"):
        other = [-p for p in CLOSE]
        assert labeling.add_vertical_barrier(events, STAMPS, close_prices=other, num_days=2) == (
            expected
        )
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="length mismatch"):
        labeling.add_vertical_barrier(events, STAMPS, CLOSE[:3], num_days=2)


def test_add_vertical_barrier_requires_a_horizon():
    with pytest.raises(ValueError, match="non-zero horizon"):
        labeling.add_vertical_barrier(STAMPS[:3], STAMPS)
    with pytest.raises(ValueError, match="non-zero horizon"):
        labeling.add_vertical_barrier(STAMPS[:3], STAMPS, None, 0, 0, 0, 0)


def test_event_on_last_bar_without_vertical_barrier_is_left_unlabelled():
    # #162: the last bar has no later bar to touch, so with no vertical barrier its t1 stays
    # None and it gets no label (it used to get t1 = t0 and a label of 0).
    first, last = STAMPS[0], STAMPS[-1]
    target = ([first, last], [0.025, 0.025])
    events = labeling.get_events(STAMPS, CLOSE, [first, last], (1.0, 1.0), *target, 0.0)
    assert [(row[0], row[1]) for row in events] == [(first, STAMPS[3]), (last, None)]

    bins = labeling.get_bins(events, STAMPS, CLOSE)
    assert [(row[0], row[3]) for row in bins] == [(first, 1)]

    labels = labeling.triple_barrier_labels(STAMPS, CLOSE, [first, last], *target, pt=1.0, sl=1.0)
    assert [row[0] for row in labels] == [first]
    meta = labeling.meta_labels(
        STAMPS, CLOSE, [first, last], *target, [(first, 1.0), (last, -1.0)], pt=1.0, sl=1.0
    )
    assert [row[0] for row in meta] == [first]


# `StabilityCriteria::default()` in crates/openquant/src/synthetic_backtesting.rs.
RUST_DEFAULT_CRITERIA = dict(
    random_walk_phi_threshold=0.97,
    min_peak_margin=0.20,
    min_surface_std=0.10,
    min_best_sharpe=0.30,
)
# The binding's own defaults before #77.
OLD_PYTHON_CRITERIA = dict(
    random_walk_phi_threshold=0.99,
    min_peak_margin=0.1,
    min_surface_std=0.05,
    min_best_sharpe=0.0,
)


def _ar1_history(phi, n=1500, seed=3, equilibrium=100.0):
    rng, p, out = random.Random(seed), equilibrium, []
    for _ in range(n):
        p = (1 - phi) * equilibrium + phi * p + rng.gauss(0, 1)
        out.append(p)
    return out


def _otr(history, **criteria):
    grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    return synthetic_bt.run_synthetic_otr_workflow(
        history,
        initial_price=97.0,
        n_paths=500,
        horizon=40,
        seed=11,
        profit_taking_grid=grid,
        stop_loss_grid=grid,
        max_holding_steps=39,
        annualization_factor=1.0,
        **criteria,
    )


def test_otr_workflow_stability_defaults_are_the_rust_defaults():
    # phi between the old (0.99) and Rust (0.97) random-walk thresholds tells them apart.
    history = _ar1_history(0.985)
    default = _otr(history)["diagnostics"]
    assert 0.97 <= default["estimated_phi"] < 0.99

    assert default == _otr(history, **RUST_DEFAULT_CRITERIA)["diagnostics"]
    assert default["no_stable_optimum"] is True
    assert _otr(history, **OLD_PYTHON_CRITERIA)["diagnostics"]["no_stable_optimum"] is False

    # Each threshold can still be overridden on its own.
    loose = _otr(
        history,
        random_walk_phi_threshold=0.999,
        min_peak_margin=0.0,
        min_surface_std=0.0,
        min_best_sharpe=-10.0,
    )
    assert loose["diagnostics"]["no_stable_optimum"] is False
