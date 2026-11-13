"""Registry identity/count integrity, independent of compiled statistics calls."""

import json

import pytest
from openquant import evaluation as ev


@pytest.mark.parametrize("mutation_source", ["input", "record_result", "trials_result"])
def test_registry_configs_are_detached_snapshots(tmp_path, mutation_source):
    config = {"model": {"lags": [1, 2]}}
    registry = ev.TrialRegistry(tmp_path / "trials.json")
    trial = registry.record(config, [0.0, 1.0, 2.0])
    exposed = {
        "input": config,
        "record_result": trial.config,
        "trials_result": registry.trials[0].config,
    }
    exposed[mutation_source]["model"]["lags"].append(99)
    stored = registry.trials[0]
    assert stored.config == {"model": {"lags": [1, 2]}}
    assert ev.config_hash(stored.config) == stored.config_hash
    assert stored == ev.TrialRegistry(registry.path).trials[0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "1", True, None])
@pytest.mark.parametrize("field", ["sharpe", "skewness", "kurtosis"])
def test_registry_rejects_invalid_persisted_moments(tmp_path, field, bad):
    registry = ev.TrialRegistry(tmp_path / "trials.json")
    registry.record({}, [0.0, 1.0, 2.0])
    payload = json.loads(registry.path.read_text())
    payload["trials"][0][field] = bad
    registry.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="malformed trial"):
        registry.reload()
    assert registry.n_trials == 1  # A failed reload must preserve the prior snapshot.


@pytest.mark.parametrize("bad", [0, 2, 3.5, True, "3", None])
def test_registry_rejects_invalid_observation_count(tmp_path, bad):
    registry = ev.TrialRegistry(tmp_path / "trials.json")
    registry.record({}, [0.0, 1.0, 2.0])
    payload = json.loads(registry.path.read_text())
    payload["trials"][0]["n_obs"] = bad
    registry.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="malformed trial"):
        registry.reload()


def test_registry_rejects_duplicate_hashes_instead_of_inflating_trial_count(tmp_path):
    registry = ev.TrialRegistry(tmp_path / "trials.json")
    registry.record({}, [0.0, 1.0, 2.0])
    payload = json.loads(registry.path.read_text())
    payload["trials"].append(payload["trials"][0].copy())
    registry.path.write_text(json.dumps(payload))
    before = registry.path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        registry.record({"other": 1}, [0.0, 1.0, 2.0])
    assert registry.path.read_bytes() == before
    assert registry.n_trials == 1


@pytest.mark.parametrize("bad", [{}, "", None, 5])
def test_registry_requires_trial_array(tmp_path, bad):
    path = tmp_path / "trials.json"
    path.write_text(json.dumps({"schema": 1, "trials": bad}))
    with pytest.raises(ValueError, match="malformed trial"):
        ev.TrialRegistry(path)


def test_registry_write_failure_preserves_disk_and_memory(tmp_path, monkeypatch):
    registry = ev.TrialRegistry(tmp_path / "trials.json")
    registry.record({"version": 1}, [0.0, 1.0, 2.0])
    before = registry.path.read_bytes()
    snapshot = registry.trials

    def fail_replace(*args):
        raise OSError("simulated disk replacement failure")

    monkeypatch.setattr(ev.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated disk"):
        registry.record({"version": 2}, [0.0, 1.0, 2.0])
    assert registry.path.read_bytes() == before
    assert registry.trials == snapshot
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trials.json"]


def test_registry_accepts_empty_and_integer_moments(tmp_path):
    path = tmp_path / "trials.json"
    path.write_text(json.dumps({"schema": 1, "trials": []}))
    registry = ev.TrialRegistry(path)
    assert registry.n_trials == 0
    registry.record({"nested": [{"key": 1}]}, [0.0, 0.0, 0.0])
    payload = json.loads(path.read_text())
    for field, value in [("sharpe", 0), ("skewness", 0), ("kurtosis", 3)]:
        payload["trials"][0][field] = value
    path.write_text(json.dumps(payload))
    registry.reload()
    registry.trials[0].config["nested"][0]["key"] = 2
    assert registry.trials[0].config == {"nested": [{"key": 1}]}
    assert registry.trials[0].sharpe == 0
