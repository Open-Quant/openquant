"""One real-data-path example, checked against independent arithmetic and time spans."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "research_journey.py"


@pytest.fixture(scope="module")
def journey():
    assert SCRIPT.is_file(), "the runnable research journey is missing"
    spec = importlib.util.spec_from_file_location("research_journey", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_bucket_frequency_has_hand_computed_values(journey):
    # Negative bucket: (1 + 3 positives) / (2 + 4 total) = 2/3.
    # Positive bucket: (1 + 0 positives) / (2 + 2 total) = 1/4.
    assert journey.fit_bucket_probabilities([-1, -1, 1], [0, 1, 0], [1, 3, 2]) == {
        "negative": pytest.approx(2 / 3),
        "nonnegative": pytest.approx(1 / 4),
    }
    assert journey.fit_bucket_probabilities([-1], [1], [0]) == {
        "negative": 0.5,
        "nonnegative": 0.5,
    }


def test_journey_is_repeatable_and_scores_only_tested_events(journey):
    result = journey.run_journey()
    assert result == journey.run_journey()
    assert result["data"]["observation_rows"] == 1024
    assert result["data"]["bar_rows"] < 1024
    assert result["data"]["event_rows"] > 50
    assert len(result["folds"]) == 4
    preds = result["predictions"]
    sample_ids = [p["sample_id"] for p in preds]
    assert len(set(sample_ids)) == len(preds)
    assert min(sample_ids) > 0  # leading training fold has no predictions
    assert len(preds) < len(result["events"])
    correct = sum(p["label"] == (p["probability"] >= 0.5) for p in preds)
    assert result["metrics"]["accuracy"] == correct / len(preds)
    assert all(0 < p["probability"] < 1 for p in preds)
    # No permissive NaN encoding in artifacts.
    json.dumps(result, allow_nan=False)


def test_training_spans_and_weights_exclude_the_test_fold(journey):
    prepared = journey.prepare_events(journey.make_observations(1024, 7))
    evaluated = journey.evaluate_events(prepared["bars"], prepared["events"])
    events = prepared["events"]
    closes = prepared["bars"]["close"].to_list()
    for fold in evaluated["folds"]:
        train = [events[i] for i in fold["train_indices"]]
        test = [events[i] for i in fold["test_indices"]]
        assert train and test
        assert max(e["end_bar"] for e in train) < min(e["start_bar"] for e in test)
        # Independent return-attribution reference: closed intervals, train events only.
        attribution = []
        for event in train:
            value = 0.0
            for bar in range(max(1, event["start_bar"]), event["end_bar"] + 1):
                concurrency = sum(e["start_bar"] <= bar <= e["end_bar"] for e in train)
                value += math.log(closes[bar] / closes[bar - 1]) / concurrency
            attribution.append(abs(value))
        total = sum(attribution)
        expected = [w * len(train) / total for w in attribution]
        assert fold["training_weights"] == pytest.approx(expected, rel=1e-10, abs=1e-10)
        assert fold["overlap_count_after_purge"] == 0


def test_later_prices_do_not_change_prior_features_or_resolved_labels(journey):
    original = journey.make_observations(1024, 7)
    cutoff = original["ts"][700]
    modified = original.with_columns(
        [
            pl.when(pl.col("ts") >= cutoff).then(pl.col(c) * 1.2).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low", "close")
        ]
    )
    before = journey.prepare_events(original)
    after = journey.prepare_events(modified)
    old_rows = [e for e in before["events"] if e["t1"] < str(cutoff)]
    new_by_start = {e["t0"]: e for e in after["events"]}
    assert old_rows
    for event in old_rows:
        assert event == new_by_start[event["t0"]]


def test_future_labels_cannot_change_first_fold_model(journey):
    prepared = journey.prepare_events(journey.make_observations(1024, 7))
    original = journey.evaluate_events(prepared["bars"], prepared["events"])
    first = original["folds"][0]
    cutoff = min(first["test_indices"])
    changed = [
        dict(e, label=1 - e["label"]) if i >= cutoff else dict(e)
        for i, e in enumerate(prepared["events"])
    ]
    alternate = journey.evaluate_events(prepared["bars"], changed)
    assert alternate["folds"][0] == first
    first_probs = [p["probability"] for p in original["predictions"] if p["split_id"] == 0]
    alternate_probs = [p["probability"] for p in alternate["predictions"] if p["split_id"] == 0]
    assert alternate_probs == first_probs


@pytest.mark.parametrize("n_rows,seed", [(0, 7), (255, 7), (1024, -1), (True, 7), (1024, 1.5)])
def test_invalid_generation_parameters_fail(journey, n_rows, seed):
    with pytest.raises(ValueError):
        journey.make_observations(n_rows, seed)


def test_insufficient_events_fail_instead_of_changing_validation(journey):
    flat = journey.make_observations(256, 7).with_columns(
        [pl.lit(100.0).alias(c) for c in ("open", "high", "low", "close")]
    )
    with pytest.raises(ValueError, match="events"):
        journey.prepare_events(flat)


def test_empty_training_fold_is_an_error(journey):
    prepared = journey.prepare_events(journey.make_observations(1024, 7))
    events = [
        dict(e, t1=prepared["events"][-1]["t1"], end_bar=prepared["events"][-1]["end_bar"])
        for e in prepared["events"]
    ]
    with pytest.raises(ValueError, match="training"):
        journey.evaluate_events(prepared["bars"], events)


def test_cli_creates_an_auditable_report_without_overwriting(journey, tmp_path):
    path = tmp_path / "report.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    args = [sys.executable, str(SCRIPT), "--out", str(path)]
    first = subprocess.run(args, capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    report = json.loads(path.read_text())
    assert report["schema_version"] == 1
    digest = report["data"]["observations_sha256"]
    assert digest.startswith("sha256:") and len(digest.removeprefix("sha256:")) == 64
    assert len(report["provenance"]["example_source_sha256"]) == 64
    assert report["predictions"] and report["limitations"]
    prior = path.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True, env=env)
    assert second.returncode != 0
    assert path.read_bytes() == prior
