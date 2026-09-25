import pytest

from openquant import streaming_hpc

# pipeline_cfg() in crates/openquant/tests/streaming_hpc.rs
PIPELINE_CFG = dict(
    bucket_volume=1000.0,
    support_buckets=10,
    lookback_events=120,
    vpin_cdf_threshold=0.99,
    hhi_threshold=0.20,
    cdf_lookback=100,
)

# snapshot rows: (timestamp_ns, price, vpin, hhi, normalized_risk_score, is_alert, vpin_cdf)
VPIN, HHI, SCORE, ALERT, VPIN_CDF = 2, 3, 4, 5, 6


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

    snapshots = report["snapshots"]
    split = int(len(events) * 0.70)
    pre = snapshots[split // 2 : split]
    post = snapshots[split : split + split // 4]

    assert _mean_present([s[VPIN] for s in post]) > _mean_present([s[VPIN] for s in pre])
    assert _mean_present([s[HHI] for s in post]) > _mean_present([s[HHI] for s in pre])
    assert report["alert_count"] > 0
    assert report["alert_count"] == sum(1 for s in snapshots if s[ALERT])
    assert not any(s[ALERT] for s in pre)
    assert report["metrics"]["processed_events"] == 4000
    # The score is >= 1 exactly when the snapshot alerts.
    for s in snapshots:
        if s[SCORE] is not None:
            assert (s[SCORE] >= 1.0) == s[ALERT]


def test_vpin_and_hhi_values_for_constant_flow():
    # Every event trades 40 buy / 60 sell on one venue, so once a 100-unit bucket has
    # filled VPIN = |40 - 60| / 100 = 0.2, and a single venue gives HHI = 1. Every VPIN
    # value ties with every other, so its CDF (ties counting half) is 0.5: a steady
    # stream is not its own extreme.
    events = [(i * 1_000_000, 100.0, 40.0, 60.0, 0) for i in range(50)]
    report = streaming_hpc.run_streaming_pipeline(
        events,
        bucket_volume=100.0,
        support_buckets=8,
        lookback_events=10,
        vpin_cdf_threshold=0.9,
        hhi_threshold=0.5,
        cdf_lookback=20,
    )

    timestamp_ns, price, vpin, hhi, _, is_alert, vpin_cdf = report["snapshots"][-1]
    assert (timestamp_ns, price) == (49_000_000, 100.0)
    assert vpin == pytest.approx(0.2, abs=1e-12)
    assert hhi == pytest.approx(1.0, abs=1e-12)
    assert vpin_cdf == 0.5
    assert is_alert is False
    assert report["alert_count"] == 0
    # VPIN exists from bucket 8 (index 7); the CDF needs 20 VPIN values (index 26).
    assert report["snapshots"][25][VPIN_CDF] is None
    assert report["snapshots"][26][VPIN_CDF] == 0.5


def test_one_vpin_cdf_threshold_serves_instruments_with_different_baselines():
    # Mirrors crates/openquant/tests/streaming_hpc.rs::
    # vpin_cdf_threshold_adapts_to_each_instruments_baseline. B's calm VPIN (0.5) is above
    # A's shock VPIN (0.4), so no raw-VPIN threshold could separate calm from shock for both.
    calm, shock = 300, 50
    for calm_flow, shock_flow in [((45.0, 55.0), (30.0, 70.0)), ((25.0, 75.0), (5.0, 95.0))]:
        events = [
            (i * 1_000, 100.0, *(calm_flow if i < calm else shock_flow), 0)
            for i in range(calm + shock)
        ]
        report = streaming_hpc.run_streaming_pipeline(
            events,
            bucket_volume=100.0,
            support_buckets=5,
            lookback_events=20,
            vpin_cdf_threshold=0.99,
            hhi_threshold=0.5,
            cdf_lookback=50,
        )
        snapshots = report["snapshots"]
        assert not any(s[ALERT] for s in snapshots[:calm])
        assert any(s[ALERT] for s in snapshots[calm:])


def test_hhi_weights_venues_by_volume():
    # Mirrors crates/openquant/tests/streaming_hpc.rs::hhi_weights_venues_by_volume_not_event_count.
    # Every 10 events: 8 one-unit trades on venue 0, one 46-unit trade on each of venues 1 and 2.
    # Volume shares 0.08 / 0.46 / 0.46 give HHI 0.4296; event counts would give 0.66.
    def row(i):
        if i % 10 == 8:
            return (i * 1_000, 100.0, 23.0, 23.0, 1)
        if i % 10 == 9:
            return (i * 1_000, 100.0, 23.0, 23.0, 2)
        return (i * 1_000, 100.0, 0.5, 0.5, 0)

    report = streaming_hpc.run_streaming_pipeline(
        [row(i) for i in range(200)],
        bucket_volume=50.0,
        support_buckets=2,
        lookback_events=10,
        vpin_cdf_threshold=0.9,
        hhi_threshold=0.5,
        cdf_lookback=10,
    )
    for s in report["snapshots"][9:]:
        assert s[HHI] == pytest.approx(0.08**2 + 2 * 0.46**2, abs=1e-12)


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
    assert report["snapshots"][-1][VPIN] is not None
    assert report["snapshots"][-1][VPIN_CDF] is not None
    assert report["snapshots"][-1][HHI] is not None
    # Ten venues rotate with identical flow, so the last 120 events give each venue a
    # tenth of the volume and HHI = 10 * 0.1^2.
    assert report["snapshots"][-1][HHI] == pytest.approx(0.1, abs=1e-9)


def test_streaming_hpc_rejects_invalid_config():
    events = streaming_hpc.generate_synthetic_flash_crash_stream(events=10)
    with pytest.raises(ValueError, match="bucket_volume"):
        streaming_hpc.run_streaming_pipeline(events, **{**PIPELINE_CFG, "bucket_volume": 0.0})
    # The threshold is a probability, and must be reachable with the given history:
    # with 20 values the CDF is at most 1 - 0.5 / 20 = 0.975.
    with pytest.raises(ValueError, match="vpin_cdf"):
        streaming_hpc.run_streaming_pipeline(
            events, **{**PIPELINE_CFG, "vpin_cdf_threshold": 1.0}
        )
    with pytest.raises(ValueError, match="unreachable"):
        streaming_hpc.run_streaming_pipeline(events, **{**PIPELINE_CFG, "cdf_lookback": 20})
    with pytest.raises(ValueError, match="cdf_lookback"):
        streaming_hpc.run_streaming_pipeline(events, **{**PIPELINE_CFG, "cdf_lookback": 1})
    # The raw-VPIN keyword is gone rather than silently reinterpreted as a CDF level.
    legacy = {k: v for k, v in PIPELINE_CFG.items() if k != "vpin_cdf_threshold"}
    with pytest.raises(TypeError):
        streaming_hpc.run_streaming_pipeline(events, vpin_threshold=0.45, **legacy)
    with pytest.raises(ValueError, match="events must be > 0"):
        streaming_hpc.generate_synthetic_flash_crash_stream(events=0)
    with pytest.raises(ValueError, match="crash_start_fraction"):
        streaming_hpc.generate_synthetic_flash_crash_stream(events=100, crash_start_fraction=1.5)
