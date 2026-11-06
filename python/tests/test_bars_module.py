from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import openquant
import polars as pl
import pytest
from polars.testing import assert_frame_equal


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
    volume = [
        80_000.0,
        85_000.0,
        90_000.0,
        88_000.0,
        95_000.0,
        300_000.0,
        92_000.0,
        98_000.0,
        99_000.0,
        97_000.0,
    ]
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


_THRESHOLD_BUILDERS = [
    pytest.param(lambda df: openquant.bars.build_tick_bars(df, ticks_per_bar=2), id="tick"),
    pytest.param(lambda df: openquant.bars.build_volume_bars(df, volume_per_bar=2.0), id="volume"),
    pytest.param(
        lambda df: openquant.bars.build_dollar_bars(df, dollar_value_per_bar=21.0), id="dollar"
    ),
]


def _mixed_symbols_frame(full_symbol: str = "A", partial_symbol: str = "B") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": ["2024-01-01 09:30:00", "2024-01-01 09:31:00", "2024-01-01 09:30:00"],
            "symbol": [full_symbol, full_symbol, partial_symbol],
            "open": [10.0, 11.0, 20.0],
            "high": [10.0, 11.0, 20.0],
            "low": [10.0, 11.0, 20.0],
            "close": [10.0, 11.0, 20.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )


def _expected_full_bar(symbol: str = "A") -> pl.DataFrame:
    # Hand-computed output for the first two observations, including canonical dtypes.
    return pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1, 9, 31)],
            "symbol": [symbol],
            "open": [10.0],
            "high": [11.0],
            "low": [10.0],
            "close": [11.0],
            "volume": [2.0],
            "adj_close": [11.0],
            "start_ts": [datetime(2024, 1, 1, 9, 30)],
            "n_obs": [2],
            "dollar_value": [21.0],
        }
    )


@pytest.mark.parametrize("build", _THRESHOLD_BUILDERS)
@pytest.mark.parametrize("symbols", [("A", "B"), ("B", "A")])
@pytest.mark.parametrize("reverse_input", [False, True])
def test_threshold_bars_drop_only_the_incomplete_symbol(
    build: Callable[[pl.DataFrame], pl.DataFrame],
    symbols: tuple[str, str],
    reverse_input: bool,
) -> None:
    frame = _mixed_symbols_frame(*symbols)
    if reverse_input:
        frame = frame.reverse()
    assert_frame_equal(build(frame), _expected_full_bar(symbols[0]))


@pytest.mark.parametrize("build", _THRESHOLD_BUILDERS)
@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_and_full_bar_results_concatenate_in_either_order(
    build: Callable[[pl.DataFrame], pl.DataFrame], empty_first: bool
) -> None:
    # Both frames cross the real Rust/Python boundary. A partial symbol must not
    # change the schema or make a multi-symbol concat depend on traversal order.
    frame = _mixed_symbols_frame()
    full = build(frame.filter(pl.col("symbol") == "A"))
    empty = build(frame.filter(pl.col("symbol") == "B"))
    frames = [empty, full] if empty_first else [full, empty]
    assert_frame_equal(pl.concat(frames, how="vertical"), _expected_full_bar())


@pytest.mark.parametrize("build", _THRESHOLD_BUILDERS)
def test_all_partial_symbols_return_the_bar_schema(
    build: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    frame = _mixed_symbols_frame().filter(pl.col("ts") == "2024-01-01 09:30:00")
    assert_frame_equal(build(frame), _expected_full_bar().clear())


@pytest.mark.parametrize(
    "build",
    [
        *_THRESHOLD_BUILDERS,
        pytest.param(lambda df: openquant.bars.build_time_bars(df, interval="1m"), id="time"),
    ],
)
@pytest.mark.parametrize("input_kind", ["empty", "all-invalid"])
def test_empty_bar_inputs_keep_the_bar_schema(
    build: Callable[[pl.DataFrame], pl.DataFrame], input_kind: str
) -> None:
    frame = _mixed_symbols_frame()
    if input_kind == "empty":
        frame = frame.clear()
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("close"))
    assert_frame_equal(build(frame), _expected_full_bar().clear())


def test_time_bars_keep_an_incomplete_symbol() -> None:
    expected = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1, 9, 31), datetime(2024, 1, 1, 9, 30)],
            "symbol": ["A", "B"],
            "open": [10.0, 20.0],
            "high": [11.0, 20.0],
            "low": [10.0, 20.0],
            "close": [11.0, 20.0],
            "volume": [2.0, 1.0],
            "adj_close": [11.0, 20.0],
            "start_ts": [datetime(2024, 1, 1, 9, 30)] * 2,
            "n_obs": [2, 1],
            "dollar_value": [21.0, 20.0],
        }
    )
    assert_frame_equal(
        openquant.bars.build_time_bars(_mixed_symbols_frame(), interval="1m"), expected
    )
