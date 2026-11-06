"""One offline research path, not a strategy or trading recommendation.

Run from the repository root after building the Python extension:
    python examples/research_journey.py --out /tmp/openquant-journey.json

See docs-site/src/content/docs/workflows/research-journey.md for the contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from openquant import bars, cross_validation, data, evaluation, filters, labeling, sample_weights

DOLLARS_PER_BAR = 200_000.0
CUSUM_THRESHOLD = 0.003
TARGET_RETURN = 0.004
HORIZON_MINUTES = 12
N_SPLITS = 5


def make_observations(n_rows: int, seed: int) -> pl.DataFrame:
    """Generate trade-like rows: each row is one price, not an aggregate candle."""
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 256:
        raise ValueError("n_rows must be an integer of at least 256")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = random.Random(seed)
    price = 100.0
    prices = []
    for _ in range(n_rows):
        price *= math.exp(rng.gauss(0.0, 0.002))
        prices.append(price)
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=i) for i in range(n_rows)],
            "symbol": ["SYNTHETIC"] * n_rows,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000.0] * n_rows,
        }
    )


def prepare_events(observations: pl.DataFrame) -> dict:
    """Use the same cleaned observations for bars, causal features and label spans."""
    clean, quality = data.clean_ohlcv(observations, return_report=True)
    if clean["symbol"].n_unique() != 1:
        raise ValueError("this example requires exactly one symbol")
    sampled = bars.build_dollar_bars(clean, dollar_value_per_bar=DOLLARS_PER_BAR)
    # The builder drops trailing partial dollar bars. Keep this completeness guard
    # explicit so a file-end close is never treated as an activity-threshold
    # observation available in a streaming run.
    sampled = sampled.filter(pl.col("dollar_value") >= DOLLARS_PER_BAR)
    timestamps = [str(ts) for ts in sampled["ts"].to_list()]
    closes = sampled["close"].to_list()
    starts = filters.cusum_filter_timestamps(closes, timestamps, CUSUM_THRESHOLD)
    vertical = labeling.add_vertical_barrier(starts, timestamps, num_minutes=HORIZON_MINUTES)
    # Missing vertical barriers mean "unbounded" to the labeling API. Exclude those
    # right-censored starts explicitly instead of silently shortening their horizon.
    eligible_starts = [start for start, _ in vertical]
    resolved = labeling.triple_barrier_events(
        timestamps,
        closes,
        eligible_starts,
        timestamps,
        [TARGET_RETURN] * len(closes),
        vertical_barrier_times=vertical,
        side_prediction=[(start, 1.0) for start in eligible_starts],
    )
    bins = {row[0]: row for row in labeling.get_bins(resolved, timestamps, closes)}
    positions = {ts: i for i, ts in enumerate(timestamps)}
    events: list[dict] = []
    for t0, t1, *_ in resolved:
        if t1 is None or t0 not in bins:
            continue
        i = positions[t0]
        # Available at the event bar's close; no future prices or fitted transforms.
        feature = math.log(closes[i] / closes[i - 1])
        events.append(
            {
                "sample_id": len(events),
                "t0": t0,
                "t1": t1,
                "start_bar": i,
                "end_bar": positions[t1],
                "feature": feature,
                "label": int(bins[t0][3]),
                "event_return": float(bins[t0][1]),
            }
        )
    if len(events) < N_SPLITS:
        raise ValueError("too few resolved events for five-fold walk-forward evaluation")
    return {
        "bars": sampled,
        "events": events,
        "quality": quality,
        "observations_sha256": data.dataset_hash(clean),
        "bars_sha256": data.dataset_hash(sampled),
        "observation_rows": clean.height,
        "cusum_event_count": len(starts),
        "excluded_without_vertical_barrier": len(starts) - len(eligible_starts),
    }


def fit_bucket_probabilities(features, labels, weights) -> dict[str, float]:
    """Educational baseline: weighted positive frequency with one pseudo-count/class.

    The feature's sign defines two fixed buckets. There is no fitted threshold,
    calibration claim, optimizer or hyperparameter search.
    """
    probabilities = {}
    for name, nonnegative in (("negative", False), ("nonnegative", True)):
        selected = [(y, w) for x, y, w in zip(features, labels, weights) if (x >= 0) == nonnegative]
        probabilities[name] = (1.0 + sum(y * w for y, w in selected)) / (
            2.0 + sum(w for _, w in selected)
        )
    return probabilities


def evaluate_events(sampled: pl.DataFrame, events: list[dict], n_splits: int = N_SPLITS) -> dict:
    """Fit only on resolved past events, weight within each fold, predict once/event."""
    splits = cross_validation.walk_forward_split_with_diagnostics(
        [e["t0"] for e in events], [e["t1"] for e in events], n_splits=n_splits
    )
    timestamps = [str(ts) for ts in sampled["ts"].to_list()]
    closes = sampled["close"].to_list()
    folds, predictions = [], []
    for split in splits:
        train_ids = split["train_indices"].tolist()
        test_ids = split["test_indices"].tolist()
        if not train_ids:
            raise ValueError("empty training fold after purging; need more resolved history")
        train = [events[i] for i in train_ids]
        # Concurrency and return attribution see training events only. Truncate the
        # price array too, so even the weight call cannot inspect a test-period price.
        last_train_bar = max(e["end_bar"] for e in train)
        weighted = sample_weights.get_weights_by_return(
            [(e["t0"], e["t1"], float(e["label"])) for e in train],
            timestamps[: last_train_bar + 1],
            closes[: last_train_bar + 1],
        )
        weight_by_start = dict(weighted)
        weights = [weight_by_start[e["t0"]] for e in train]
        probabilities = fit_bucket_probabilities(
            [e["feature"] for e in train], [e["label"] for e in train], weights
        )
        # A second, feature-free training baseline makes class imbalance visible.
        constant_probability = (1.0 + sum(e["label"] * w for e, w in zip(train, weights))) / (
            2.0 + sum(weights)
        )
        folds.append(
            {
                "split_id": int(split["split_id"]),
                "train_indices": train_ids,
                "test_indices": test_ids,
                "purged_indices": split["purged_indices"].tolist(),
                "training_weights": weights,
                "bucket_probabilities": probabilities,
                "constant_probability": constant_probability,
                "overlap_count_after_purge": int(split["overlap_count_after_purge"]),
            }
        )
        for i in test_ids:
            e = events[i]
            predictions.append(
                {
                    "sample_id": i,
                    "split_id": int(split["split_id"]),
                    "t0": e["t0"],
                    "t1": e["t1"],
                    "label": e["label"],
                    "probability": probabilities[
                        "nonnegative" if e["feature"] >= 0 else "negative"
                    ],
                    "constant_probability": constant_probability,
                }
            )
    return {
        "folds": folds,
        "predictions": predictions,
        "metrics": evaluation.meta_label_metrics(
            [p["label"] for p in predictions], [p["probability"] for p in predictions]
        ),
        "baseline_metrics": evaluation.meta_label_metrics(
            [p["label"] for p in predictions], [p["constant_probability"] for p in predictions]
        ),
    }


def run_journey(n_rows: int = 1024, seed: int = 7) -> dict:
    """Return a JSON-safe report for one fixed, synthetic experimental procedure."""
    prepared = prepare_events(make_observations(n_rows, seed))
    result = evaluate_events(prepared["bars"], prepared["events"])
    root = Path(__file__).resolve().parents[1]
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "schema_version": 1,
        "purpose": "synthetic API integration and validation example, not trading evidence",
        "config": {
            "n_rows": n_rows,
            "seed": seed,
            "dollars_per_bar": DOLLARS_PER_BAR,
            "cusum_threshold": CUSUM_THRESHOLD,
            "target_return": TARGET_RETURN,
            "horizon_minutes": HORIZON_MINUTES,
            "n_splits": N_SPLITS,
            "primary_side": "always long",
            "decision_threshold": 0.5,
        },
        "provenance": {
            "git_commit": git_commit,
            "example_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
            "polars": importlib.metadata.version("polars"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "data": {k: v for k, v in prepared.items() if k not in {"bars", "events"}}
        | {"bar_rows": prepared["bars"].height, "event_rows": len(prepared["events"])},
        "events": prepared["events"],
        **result,
        "limitations": [
            "Synthetic zero-drift log-return path; not representative market data.",
            "Frequency scores are illustrative and not calibrated probabilities.",
            "Labels use close-to-close gross returns; fees, fills and slippage are absent.",
            "Classification metrics are not strategy or portfolio returns.",
            "Overlapping test labels are dependent; no iid confidence interval is claimed.",
            "Later folds may train on earlier resolved test events; no untouched final holdout.",
            "Purging addresses label overlap, not every possible source of information leakage.",
        ],
    }


def write_report(report: dict, path: Path) -> None:
    """Create a new report without destroying an earlier run."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        report = run_journey(n_rows=args.rows, seed=args.seed)
        write_report(report, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"research journey: {exc}\n")
    print(
        f"Wrote {args.out}: {report['data']['bar_rows']} bars, "
        f"{report['data']['event_rows']} resolved events, "
        f"{len(report['predictions'])} walk-forward predictions."
    )
    print("Synthetic classification example only. Inspect limitations before interpreting scores.")


if __name__ == "__main__":
    main()
