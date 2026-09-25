"""Invalid arguments raise ``ValueError``; a Rust panic (``PanicException``) is a bug (#36).

``PanicException`` derives from ``BaseException``, so ``pytest.raises(ValueError)`` does not
accept it: each of these failed before the core returned ``Err`` for the input.
"""

import pytest
from openquant._core import backtest_stats, bars, bet_sizing, ef3m, microstructural, sampling

MOMENTS = [0.7, 2.6, 0.4, 25.0, -59.8]
T1 = ["2024-01-01 00:00:00", "2024-01-02 00:00:00"]
T2 = ["2024-01-03 00:00:00", "2024-01-04 00:00:00"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: backtest_stats.deflated_sharpe_ratio(1.14, [0.5], 250, 0.0, 3.0),
        lambda: backtest_stats.deflated_sharpe_ratio(1.14, [0.5, 1.0], 250, 0.0, 3.0, True),
        lambda: backtest_stats.minimum_track_record_length(1.14, 1.0, 0.0, 3.0, 1.5),
        lambda: bet_sizing.get_target_pos_power(2.0, 3.0, 1.0, 10.0),
        lambda: bet_sizing.bet_size_reserve_full([], [], [], 1, 1e-5, 10, False),
        lambda: bars.build_tick_bars(T1, [1.0, 2.0], [1.0, 1.0], 0),
        lambda: ef3m.fit_m2n([0.7], epsilon=1e-2, max_iter=10),
        lambda: ef3m.fit_m2n(MOMENTS, epsilon=1e-2, max_iter=10, variant=3),
        lambda: microstructural.quantile_mapping([1.0, float("nan")], 2),
        lambda: microstructural.get_roll_impact([1.0, 2.0, 3.0], [1.0], 2),
        lambda: microstructural.get_plug_in_entropy("", 1),
        lambda: sampling.seq_bootstrap([[1, 0], [1]], None, None),
        lambda: sampling.seq_bootstrap([[1, 0], [1, 1]], 2, [7]),
        lambda: sampling.get_ind_matrix([(5, 2)], [0, 1, 2]),
    ],
    ids=[
        "dsr_single_estimate",
        "dsr_single_trial",
        "mtrl_alpha_above_one",
        "target_pos_power_divergence",
        "reserve_full_empty",
        "tick_bars_zero_threshold",
        "fit_m2n_short_moments",
        "fit_m2n_unknown_variant",
        "quantile_mapping_nan",
        "roll_impact_length_mismatch",
        "plug_in_entropy_empty_message",
        "seq_bootstrap_ragged",
        "seq_bootstrap_warmup_out_of_range",
        "ind_matrix_end_before_start",
    ],
)
def test_invalid_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_error_message_names_the_argument():
    with pytest.raises(ValueError, match=r"'low' has length 1, expected 3"):
        microstructural.get_corwin_schultz_estimator([1.0, 2.0, 3.0], [1.0], 2)


def test_fit_m2n_default_variant_is_one_the_core_accepts():
    # The binding used to default to variant=4, which the core rejects; every error was then
    # discarded and the default call returned [] for any input.
    #
    # The fit starts from a random p_1 and about one run in twelve finds nothing better than its
    # starting error, so a single call may legitimately return []. Twenty empty runs in a row
    # (p ~ 1e-21) means the default is broken again.
    runs = [ef3m.fit_m2n(MOMENTS, epsilon=1e-2, max_iter=1000) for _ in range(20)]
    fitted = [rows[0] for rows in runs if rows]
    assert fitted
    assert all(0.0 <= row[4] <= 1.0 for row in fitted)


def test_entropy_accepts_every_letter_the_encoder_emits():
    # quantile_mapping uses letters up to U+00FF. Those above U+007F are two bytes in UTF-8 and
    # the entropy functions sliced the message by byte.
    values = [float((i * 37) % 400) for i in range(400)]
    message = microstructural.encode_array(values, microstructural.quantile_mapping(values, 200))
    assert not message.isascii()

    assert microstructural.get_shannon_entropy(message) > 0.0
    assert microstructural.get_plug_in_entropy(message, 2) > 0.0
    assert 0.0 < microstructural.get_lempel_ziv_entropy(message) <= 1.0
    assert microstructural.get_konto_entropy(message, 10) > 0.0
