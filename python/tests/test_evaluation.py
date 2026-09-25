"""`openquant.evaluation` against hand-computed values.

The returns series is `fixtures/evaluation_returns.csv`: 120 synthetic daily returns
written by `fixtures/make_evaluation_returns.py` (standard-library `random.Random(27)`,
rounded to six decimals). The expected numbers below were computed from that file with the
standard library alone — `statistics.NormalDist` for Z and Z⁻¹, no openquant, no statrs —
by `_hand_*` in this module, straight from the formulas in Bailey and López de Prado
(2012, 2014) and AFML §14.7. They are also written out as literals so that a change to the
hand computation cannot silently move both sides at once.
"""

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import pytest
from openquant import evaluation as ev

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "evaluation_returns.csv"
EULER_GAMMA = 0.5772156649015329
Z = NormalDist()


def _returns():
    with FIXTURE.open(newline="") as fh:
        return [float(row["return"]) for row in csv.DictReader(fh)]


RETURNS = _returns()


# --- the hand computation ---------------------------------------------------------------


def _hand_moments(r):
    n = len(r)
    mu = sum(r) / n
    m2, m3, m4 = (sum((x - mu) ** p for x in r) / n for p in (2, 3, 4))
    sd = math.sqrt(sum((x - mu) ** 2 for x in r) / (n - 1))
    return n, mu / sd, m3 / m2**1.5, m4 / m2**2


def _hand_psr(r, benchmark):
    n, sr, g3, g4 = _hand_moments(r)
    return Z.cdf(
        (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    )


def _hand_sr0(sd, n_trials):
    return sd * (
        (1 - EULER_GAMMA) * Z.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * Z.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def _hand_mintrl(r, benchmark, alpha):
    _, sr, g3, g4 = _hand_moments(r)
    return 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr**2) * (Z.inv_cdf(1 - alpha) / (sr - benchmark)) ** 2


def _pop_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# Seven rival configurations' per-period Sharpe ratios; the fixture's own is the eighth.
OTHER_TRIALS = [0.05, -0.02, 0.11, 0.03, 0.08, -0.04, 0.01]

# Literal hand-computed values for the fixture (see module docstring).
HAND = {
    "n_obs": 120,
    "sharpe": 0.12270454695517263,
    "skewness": -0.4754168553612303,
    "kurtosis": 3.6971835513701903,
    "psr_0": 0.9023293346519098,
    "psr_005": 0.7785405251665566,
    "trial_sd": 0.055199190472444204,
    "sr0_8_trials": 0.0805363959626879,
    "dsr_8_trials": 0.671844990229503,
    "sr0_100_trials_sd_005": 0.12653014466008425,
    "dsr_100_trials_sd_005": 0.48389799357774277,
    "mintrl_0_005": 193.00076753193,
    "mintrl_005_005": 547.8914481619129,
    "mintrl_0_010": 117.55223936848627,
}


def test_hand_computation_matches_its_literals():
    n, sr, g3, g4 = _hand_moments(RETURNS)
    assert n == HAND["n_obs"]
    assert sr == pytest.approx(HAND["sharpe"], rel=1e-12)
    assert g3 == pytest.approx(HAND["skewness"], rel=1e-12)
    assert g4 == pytest.approx(HAND["kurtosis"], rel=1e-12)
    assert _hand_psr(RETURNS, 0.0) == pytest.approx(HAND["psr_0"], rel=1e-12)
    trials = OTHER_TRIALS + [sr]
    assert _pop_std(trials) == pytest.approx(HAND["trial_sd"], rel=1e-12)
    assert _hand_sr0(_pop_std(trials), 8) == pytest.approx(HAND["sr0_8_trials"], rel=1e-12)
    assert _hand_mintrl(RETURNS, 0.0, 0.05) == pytest.approx(HAND["mintrl_0_005"], rel=1e-12)


# --- PSR, DSR, MinTRL ---------------------------------------------------------------------


def test_return_moments():
    m = ev.return_moments(RETURNS)
    assert m.n_obs == HAND["n_obs"]
    assert m.sharpe == pytest.approx(HAND["sharpe"], rel=1e-12)
    assert m.skewness == pytest.approx(HAND["skewness"], rel=1e-12)
    assert m.kurtosis == pytest.approx(HAND["kurtosis"], rel=1e-12)
    assert m.sharpe == pytest.approx(m.mean / m.std, rel=1e-12)


@pytest.mark.parametrize(("benchmark", "key"), [(0.0, "psr_0"), (0.05, "psr_005")])
def test_probabilistic_sharpe_ratio(benchmark, key):
    assert ev.probabilistic_sharpe_ratio(RETURNS, benchmark) == pytest.approx(HAND[key], abs=1e-9)


def test_expected_max_sharpe():
    assert ev.expected_max_sharpe(8, HAND["trial_sd"]) == pytest.approx(
        HAND["sr0_8_trials"], abs=1e-9
    )
    assert ev.expected_max_sharpe(100, 0.05) == pytest.approx(
        HAND["sr0_100_trials_sd_005"], abs=1e-9
    )
    # No dispersion across trials: luck cannot pick a winner.
    assert ev.expected_max_sharpe(10, 0.0) == 0.0


def test_deflated_sharpe_ratio_from_trial_sharpes():
    trials = OTHER_TRIALS + [HAND["sharpe"]]
    dsr = ev.deflated_sharpe_ratio(RETURNS, trials)
    assert dsr == pytest.approx(HAND["dsr_8_trials"], abs=1e-9)
    assert dsr < ev.probabilistic_sharpe_ratio(RETURNS)


def test_deflated_sharpe_ratio_from_count_and_std():
    dsr = ev.deflated_sharpe_ratio(RETURNS, n_trials=100, sharpe_std=0.05)
    assert dsr == pytest.approx(HAND["dsr_100_trials_sd_005"], abs=1e-9)


@pytest.mark.parametrize(
    ("benchmark", "alpha", "key"),
    [(0.0, 0.05, "mintrl_0_005"), (0.05, 0.05, "mintrl_005_005"), (0.0, 0.10, "mintrl_0_010")],
)
def test_minimum_track_record_length(benchmark, alpha, key):
    assert ev.minimum_track_record_length(RETURNS, benchmark, alpha) == pytest.approx(
        HAND[key], rel=1e-9
    )


def test_minimum_track_record_length_is_infinite_at_or_below_the_benchmark():
    assert ev.minimum_track_record_length(RETURNS, HAND["sharpe"] + 0.01) == math.inf
    assert ev.minimum_track_record_length([-r for r in RETURNS], 0.0) == math.inf


def test_mintrl_and_psr_agree():
    # At T = MinTRL observations, PSR reaches exactly 1 - alpha (same SR and moments).
    _, sr, g3, g4 = _hand_moments(RETURNS)
    t = ev.minimum_track_record_length(RETURNS, 0.0, 0.05)
    z = sr * math.sqrt(t - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert Z.cdf(z) == pytest.approx(0.95, abs=1e-12)


# --- trial registry -----------------------------------------------------------------------


def _shifted(shift):
    return [r + shift for r in RETURNS]


def test_registry_persists_across_runs(tmp_path):
    path = tmp_path / "trials.json"
    first_run = ev.TrialRegistry(path)
    assert first_run.n_trials == 0 and not path.exists()
    first_run.record({"lookback": 10, "threshold": 0.5}, _shifted(-0.001))
    first_run.record({"lookback": 20, "threshold": 0.5}, _shifted(0.0005))

    second_run = ev.TrialRegistry(path)  # a new process would start here
    assert second_run.n_trials == 2
    assert [t.config for t in second_run.trials] == [
        {"lookback": 10, "threshold": 0.5},
        {"lookback": 20, "threshold": 0.5},
    ]
    second_run.record({"lookback": 40, "threshold": 0.5}, RETURNS)
    assert second_run.n_trials == 3
    assert ev.TrialRegistry(path).n_trials == 3

    # The first object sees the other run's trial after reloading, or on its next write.
    first_run.reload()
    assert first_run.n_trials == 3
    first_run.record({"lookback": 80, "threshold": 0.5}, _shifted(0.001))
    assert ev.TrialRegistry(path).n_trials == 4


def test_registry_entry_fields(tmp_path):
    reg = ev.TrialRegistry(tmp_path / "trials.json")
    trial = reg.record({"b": 2, "a": 1}, RETURNS)
    assert trial.config_hash == ev.config_hash({"a": 1, "b": 2})
    assert len(trial.config_hash) == 64
    assert trial.sharpe == pytest.approx(HAND["sharpe"], rel=1e-12)
    assert trial.n_obs == HAND["n_obs"]
    assert trial.skewness == pytest.approx(HAND["skewness"], rel=1e-12)
    assert trial.kurtosis == pytest.approx(HAND["kurtosis"], rel=1e-12)
    assert trial.timestamp.endswith("+00:00")

    on_disk = json.loads((tmp_path / "trials.json").read_text())
    assert on_disk["schema"] == ev.REGISTRY_SCHEMA
    assert set(on_disk["trials"][0]) == {
        "config_hash",
        "timestamp",
        "sharpe",
        "n_obs",
        "skewness",
        "kurtosis",
        "config",
    }


def test_rerecording_a_config_does_not_add_a_trial(tmp_path):
    path = tmp_path / "trials.json"
    ev.TrialRegistry(path).record({"lookback": 10}, _shifted(-0.001))
    ev.TrialRegistry(path).record({"lookback": 20}, RETURNS)
    reg = ev.TrialRegistry(path)
    reg.record({"lookback": 10}, _shifted(0.002))  # same config, re-run
    assert reg.n_trials == 2
    assert [t.config for t in reg.trials] == [{"lookback": 20}, {"lookback": 10}]
    assert reg.trials[1].sharpe == pytest.approx(ev.return_moments(_shifted(0.002)).sharpe)


def test_registry_dsr_uses_the_registry_count(tmp_path):
    path = tmp_path / "trials.json"
    # Seven rival configurations recorded in one run, from returns whose Sharpe ratios are
    # the OTHER_TRIALS values; the evaluated configuration is recorded in a second run.
    first = ev.TrialRegistry(path)
    n, sr, *_ = _hand_moments(RETURNS)
    mean = sum(RETURNS) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in RETURNS) / (n - 1))
    for i, target in enumerate(OTHER_TRIALS):
        first.record({"variant": i}, [r - mean + target * sd for r in RETURNS])
    second = ev.TrialRegistry(path)
    second.record({"variant": "chosen"}, RETURNS)

    assert second.n_trials == 8
    assert second.sharpe_std() == pytest.approx(HAND["trial_sd"], abs=1e-12)
    assert second.expected_max_sharpe() == pytest.approx(HAND["sr0_8_trials"], abs=1e-9)
    assert second.deflated_sharpe_ratio(RETURNS) == pytest.approx(HAND["dsr_8_trials"], abs=1e-9)

    # One more trial with the same dispersion raises the hurdle and lowers the DSR.
    third = ev.TrialRegistry(path)
    third.record({"variant": "extra"}, [r - mean + 0.04 * sd for r in RETURNS])
    assert third.n_trials == 9
    assert third.deflated_sharpe_ratio(RETURNS) == pytest.approx(
        _hand_psr(RETURNS, _hand_sr0(_pop_std(OTHER_TRIALS + [sr, 0.04]), 9)), abs=1e-9
    )


def test_registry_needs_two_trials_to_deflate(tmp_path):
    reg = ev.TrialRegistry(tmp_path / "trials.json")
    with pytest.raises(ValueError, match="at least 2"):
        reg.deflated_sharpe_ratio(RETURNS)
    reg.record({"only": 1}, RETURNS)
    with pytest.raises(ValueError, match="at least 2"):
        reg.deflated_sharpe_ratio(RETURNS)


def test_registry_writes_leave_no_temporary_files(tmp_path):
    reg = ev.TrialRegistry(tmp_path / "nested" / "trials.json")
    for i in range(3):
        reg.record({"i": i}, _shifted(i * 0.0001))
    assert sorted(p.name for p in (tmp_path / "nested").iterdir()) == ["trials.json"]


def test_registry_rejects_a_foreign_file(tmp_path):
    bad = tmp_path / "trials.json"
    bad.write_text("not json")
    with pytest.raises(ValueError, match="not a trial registry"):
        ev.TrialRegistry(bad)
    bad.write_text(json.dumps({"schema": 99, "trials": []}))
    with pytest.raises(ValueError, match="schema-1"):
        ev.TrialRegistry(bad)


def test_registry_rejects_unserialisable_config(tmp_path):
    reg = ev.TrialRegistry(tmp_path / "trials.json")
    with pytest.raises(TypeError, match="JSON-serialisable"):
        reg.record({"model": object()}, RETURNS)
    assert reg.n_trials == 0 and not (tmp_path / "trials.json").exists()


# --- meta-labeling and strategy risk ------------------------------------------------------


def test_meta_label_metrics_hand_counts():
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    probs = [0.9, 0.8, 0.3, 0.6, 0.7, 0.2, 0.1, 0.4, 0.55, 0.45]
    out = ev.meta_label_metrics(labels, probs)
    # Act on 0.9, 0.8, 0.6 (label 1) and 0.7, 0.55 (label 0): TP 3, FP 2, FN 2, TN 3.
    assert (out["true_positives"], out["false_positives"]) == (3, 2)
    assert (out["false_negatives"], out["true_negatives"]) == (2, 3)
    assert out["precision"] == pytest.approx(0.6)
    assert out["recall"] == pytest.approx(0.6)
    assert out["f1"] == pytest.approx(0.6)
    assert out["accuracy"] == pytest.approx(0.6)
    # The primary model takes all ten bets, five of them profitable.
    assert out["primary_precision"] == pytest.approx(0.5)
    assert out["primary_recall"] == 1.0
    assert out["primary_f1"] == pytest.approx(2 * 0.5 / 1.5)


def test_meta_label_metrics_threshold_and_empty_selection():
    labels = [1, 0, 1, 0]
    out = ev.meta_label_metrics(labels, [0.6, 0.6, 0.6, 0.6], threshold=0.7)
    assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["f1"] == 0.0
    decisions = ev.meta_label_metrics(labels, [1, 0, 1, 0])
    assert decisions["precision"] == 1.0 and decisions["f1"] == 1.0


def test_strategy_failure_probability_matches_strategy_risk():
    from openquant import strategy_risk

    bets = [0.01 if i % 10 < 7 else -0.02 for i in range(520)]
    kwargs = dict(years_elapsed=2.0, target_sharpe=1.0, investor_horizon_years=2.0, seed=5)
    report = ev.strategy_failure_probability(bets, **kwargs)
    direct = strategy_risk.estimate_strategy_failure_probability(bets, **kwargs)
    assert report["failure_probability"] == direct["empirical_failure_probability"]
    assert report["implied_precision_threshold"] == direct["implied_precision_threshold"]
    assert 0.0 <= report["failure_probability"] <= 1.0


# --- input validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returns", "error", "match"),
    [
        ([0.01, 0.02], ValueError, "at least 3"),
        ([0.01, float("nan"), 0.02], ValueError, "finite"),
        ([0.01, float("inf"), 0.02], ValueError, "finite"),
        ([0.01, 0.01, 0.01], ValueError, "constant"),
        ("0.01,0.02,0.03", TypeError, "not a string"),
        ([0.01, "x", 0.02], TypeError, "sequence of numbers"),
    ],
)
def test_returns_validation(returns, error, match):
    for fn in (ev.return_moments, ev.probabilistic_sharpe_ratio, ev.minimum_track_record_length):
        with pytest.raises(error, match=match):
            fn(returns)


def test_parameter_validation():
    with pytest.raises(ValueError, match="alpha"):
        ev.minimum_track_record_length(RETURNS, 0.0, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        ev.minimum_track_record_length(RETURNS, 0.0, 1.0)
    with pytest.raises(ValueError, match="benchmark_sr"):
        ev.probabilistic_sharpe_ratio(RETURNS, float("nan"))
    with pytest.raises(ValueError, match="n_trials must be at least 2"):
        ev.expected_max_sharpe(1, 0.05)
    with pytest.raises(TypeError, match="n_trials must be an int"):
        ev.expected_max_sharpe(10.0, 0.05)
    with pytest.raises(ValueError, match="sharpe_std"):
        ev.expected_max_sharpe(10, -0.1)
    with pytest.raises(ValueError, match="at least 2 trials"):
        ev.deflated_sharpe_ratio(RETURNS, [0.1])
    with pytest.raises(ValueError, match="not both"):
        ev.deflated_sharpe_ratio(RETURNS, [0.1, 0.2], n_trials=5, sharpe_std=0.1)
    with pytest.raises(ValueError, match="both n_trials and sharpe_std"):
        ev.deflated_sharpe_ratio(RETURNS, n_trials=5)
    with pytest.raises(ValueError, match="differ in length"):
        ev.meta_label_metrics([1, 0], [0.5])
    with pytest.raises(ValueError, match="0 or 1"):
        ev.meta_label_metrics([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ev.meta_label_metrics([1, 0], [0.5, 1.5])
    with pytest.raises(ValueError, match="empty"):
        ev.meta_label_metrics([], [])
    with pytest.raises(ValueError, match="threshold"):
        ev.meta_label_metrics([1, 0], [0.5, 0.5], threshold=0.0)
