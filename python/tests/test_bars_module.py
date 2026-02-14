from __future__ import annotations

import polars as pl

import openquant


def _base_frame() -> pl.DataFrame:
    ts = [
        "2024-01-01 09:30:00",
        "2024-01-01 09:31:00",
        "2024-01-01 09:33:00",
        "2024-01-01 09:34:00",
        "2024-01-01 09:35:00",
        "2024-01-01 09:36:00",
        "2024-01-01 09:37:00",
        "2024-01-01 09:39:00",
        "2024-01-01 09:40:00",
        "2024-01-01 09:41:00",
    ]
    close = [100.0, 100.4, 100.1, 100.8, 100.7, 101.2, 101.0, 101.5, 101.4, 101.7]
    volume = [80_000.0, 85_000.0, 90_000.0, 88_000.0, 95_000.0, 300_000.0, 92_000.0, 98_000.0, 99_000.0, 97_000.0]
    return pl.DataFrame(
        {
            "ts": ts,
            "symbol": ["SPY"] * len(ts),
            "open": close,
            "high": [x * 1.001 for x in close],
            "low": [x * 0.999 for x in close],
            "close": close,
            "volume": volume,
            "adj_close": close,
        }
    )


def _assert_bar_invariants(df: pl.DataFrame) -> None:
    assert df.height > 0
    assert df.select((pl.col("high") >= pl.col("open")).all()).item()
    assert df.select((pl.col("high") >= pl.col("close")).all()).item()
    assert df.select((pl.col("low") <= pl.col("open")).all()).item()
    assert df.select((pl.col("low") <= pl.col("close")).all()).item()
    assert df.select((pl.col("n_obs") >= 1).all()).item()
    assert df.select((pl.col("ts") >= pl.col("start_ts")).all()).item()


def test_time_tick_volume_dollar_bars_are_deterministic():
    base = _base_frame()
    builders = [
        lambda x: openquant.bars.build_time_bars(x, interval="5m"),
        lambda x: openquant.bars.build_tick_bars(x, ticks_per_bar=3),
        lambda x: openquant.bars.build_volume_bars(x, volume_per_bar=250_000),
        lambda x: openquant.bars.build_dollar_bars(x, dollar_value_per_bar=25_000_000),
    ]
    for build in builders:
        a = build(base)
        b = build(base)
        assert a.equals(b)


def test_bar_outputs_monotone_and_invariant():
    base = _base_frame()
    for out in (
        openquant.bars.build_time_bars(base, interval="5m"),
        openquant.bars.build_tick_bars(base, ticks_per_bar=2),
        openquant.bars.build_volume_bars(base, volume_per_bar=200_000),
        openquant.bars.build_dollar_bars(base, dollar_value_per_bar=20_000_000),
    ):
        _assert_bar_invariants(out)
        ts = out["ts"].to_list()
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


def test_sparse_intervals_and_outlier_volume_edge_cases():
    base = _base_frame()
    sparse = base.filter(pl.col("ts") != pl.lit("2024-01-01 09:34:00"))
    out = openquant.bars.build_time_bars(sparse, interval="5m")
    assert out.height >= 2
    _assert_bar_invariants(out)

    # One outlier volume observation should not break grouping.
    outlier = base.with_columns(
        pl.when(pl.col("ts") == pl.lit("2024-01-01 09:36:00"))
        .then(pl.lit(5_000_000.0))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    vol_bars = openquant.bars.build_volume_bars(outlier, volume_per_bar=300_000)
    _assert_bar_invariants(vol_bars)
    assert vol_bars.height >= 2
