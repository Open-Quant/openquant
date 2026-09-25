import pytest
from openquant import streaming_hpc

# pipeline_cfg() in crates/openquant/tests/streaming_hpc.rs
PIPELINE_CFG = dict(
    bucket_volume=1000.0,
    support_buckets=10,
    lookback_events=120,
    vpin_threshold=0.45,
    hhi_threshold=0.30,
)


def _mean_present(values):
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept)


def test_flash_crash_segment_raises_early_warning_metrics():
    # Mirrors crates/openquant/tests/streaming_hpc.rs::
    # flash_crash_segment_raises_early_warning_metrics
    events = streaming_hpc.generate_synthetic_flash_crash_stream(
        events=4000, crash_start_fraction=0.70, calm_venues=8, shock_venue=0
    )
    report = streaming_hpc.run_streaming_pipeline(events, **PIPELINE_CFG)

    # snapshot rows: (timestamp_ns, price, vpin, hhi, normalized_risk_score, is_alert)
    snapshots = report["snapshots"]
    split = int(len(events) * 0.70)
    pre = snapshots[split // 2 : split]
    post = snapshots[split : split + split // 4]

    assert _mean_present([s[2] for s in post]) > _mean_present([s[2] for s in pre])
    assert _mean_present([s[3] for s in post]) > _mean_present([s[3] for s in pre])
    assert report["alert_count"] > 0
    assert report["alert_count"] == sum(1 for s in snapshots if s[5])
    assert report["metrics"]["processed_events"] == 4000


def test_vpin_and_hhi_values_for_constant_flow():
    # Every event trades 40 buy / 60 sell on one venue, so once a 100-unit bucket has
    # filled VPIN = |40 - 60| / 100 = 0.2, and a single venue gives HHI = 1.
    events = [(i * 1_000_000, 100.0, 40.0, 60.0, 0) for i in range(50)]
    report = streaming_hpc.run_streaming_pipeline(
        events,
        bucket_volume=100.0,
        support_buckets=8,
        lookback_events=10,
        vpin_threshold=0.45,
        hhi_threshold=2.0,
    )

    timestamp_ns, price, vpin, hhi, _, is_alert = report["snapshots"][-1]
    assert (timestamp_ns, price) == (49_000_000, 100.0)
    assert vpin == pytest.approx(0.2, abs=1e-12)
    assert hhi == pytest.approx(1.0, abs=1e-12)
    assert is_alert is False
    assert report["alert_count"] == 0


def test_supports_large_synthetic_stream_incrementally():
    # Mirrors crates/openquant/tests/streaming_hpc.rs::
    # supports_large_synthetic_stream_incrementally
    events = []
    price = 100.0
    for i in range(50_000):
        if i % 5000 >= 4200:
            buy, sell, drift = 90.0, 280.0, -0.0012
        else:
            buy, sell, drift = 140.0, 150.0, 0.00005
        price *= 1.0 + drift
        events.append((i * 500_000, price, buy, sell, i % 10))

    report = streaming_hpc.run_streaming_pipeline(events, **PIPELINE_CFG)

    assert report["metrics"]["processed_events"] == 50_000
    assert report["metrics"]["events_per_sec"] > 0.0
    assert report["snapshots"][-1][2] is not None
    assert report["snapshots"][-1][3] is not None
    # Ten venues share the flow equally, so HHI = 10 * 0.1^2.
    assert report["snapshots"][-1][3] == pytest.approx(0.1, abs=1e-9)


def test_streaming_hpc_rejects_invalid_config():
    events = streaming_hpc.generate_synthetic_flash_crash_stream(events=10)
    with pytest.raises(ValueError, match="bucket_volume"):
        streaming_hpc.run_streaming_pipeline(events, **{**PIPELINE_CFG, "bucket_volume": 0.0})
    with pytest.raises(ValueError, match="events must be > 0"):
        streaming_hpc.generate_synthetic_flash_crash_stream(events=0)
    with pytest.raises(ValueError, match="crash_start_fraction"):
        streaming_hpc.generate_synthetic_flash_crash_stream(events=100, crash_start_fraction=1.5)
