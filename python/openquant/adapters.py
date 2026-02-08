from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import polars as pl


def _validate_equal_length(name_a: str, a: Sequence[object], name_b: str, b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a}/{name_b} length mismatch: {len(a)} vs {len(b)}")


def to_polars_signal_frame(
    timestamps: Sequence[str],
    signal: Sequence[float],
    side: Sequence[float] | None = None,
    symbol: str | None = None,
) -> pl.DataFrame:
    _validate_equal_length("timestamps", timestamps, "signal", signal)
    data: dict[str, object] = {"ts": list(timestamps), "signal": list(signal)}
    if side is not None:
        _validate_equal_length("timestamps", timestamps, "side", side)
        data["side"] = list(side)
    if symbol is not None:
        data["symbol"] = [symbol] * len(timestamps)
    return pl.DataFrame(data).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))


def to_polars_event_frame(
    starts: Sequence[str],
    ends: Sequence[str],
    probs: Sequence[float],
    sides: Sequence[float] | None = None,
    labels: Sequence[int] | None = None,
) -> pl.DataFrame:
    _validate_equal_length("starts", starts, "ends", ends)
    _validate_equal_length("starts", starts, "probs", probs)
    data: dict[str, object] = {"start_ts": list(starts), "end_ts": list(ends), "prob": list(probs)}
    if sides is not None:
        _validate_equal_length("starts", starts, "sides", sides)
        data["side"] = list(sides)
    if labels is not None:
        _validate_equal_length("starts", starts, "labels", labels)
        data["label"] = list(labels)
    return pl.DataFrame(data).with_columns(
        pl.col("start_ts").str.strptime(pl.Datetime, strict=False),
        pl.col("end_ts").str.strptime(pl.Datetime, strict=False),
    )


def to_polars_indicator_matrix(
    ind_mat: Sequence[Sequence[int]],
    bar_index: Sequence[int] | None = None,
    label_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    if not ind_mat:
        return pl.DataFrame({"bar_index": []})
    width = len(ind_mat[0])
    if any(len(row) != width for row in ind_mat):
        raise ValueError("ind_mat must be rectangular")
    if label_names is None:
        label_names = [f"label_{i}" for i in range(width)]
    if len(label_names) != width:
        raise ValueError(f"label_names length mismatch: expected {width}, got {len(label_names)}")
    if bar_index is None:
        bar_index = list(range(len(ind_mat)))
    _validate_equal_length("bar_index", bar_index, "ind_mat_rows", ind_mat)

    data: dict[str, object] = {"bar_index": list(bar_index)}
    for j, name in enumerate(label_names):
        data[name] = [int(row[j]) for row in ind_mat]
    return pl.DataFrame(data)


def to_polars_weights_frame(
    asset_names: Sequence[str],
    weights: Sequence[float],
    as_of: str | None = None,
) -> pl.DataFrame:
    _validate_equal_length("asset_names", asset_names, "weights", weights)
    data: dict[str, object] = {"asset": list(asset_names), "weight": list(weights)}
    if as_of is not None:
        data["as_of"] = [as_of] * len(asset_names)
    df = pl.DataFrame(data)
    if as_of is not None:
        df = df.with_columns(pl.col("as_of").str.strptime(pl.Datetime, strict=False))
    return df


def to_polars_frontier_frame(
    volatility: Sequence[float],
    returns: Sequence[float],
    sharpe: Sequence[float] | None = None,
    point_ids: Sequence[str] | None = None,
) -> pl.DataFrame:
    _validate_equal_length("volatility", volatility, "returns", returns)
    n = len(volatility)
    data: dict[str, object] = {"volatility": list(volatility), "return": list(returns)}
    if sharpe is not None:
        _validate_equal_length("volatility", volatility, "sharpe", sharpe)
        data["sharpe"] = list(sharpe)
    if point_ids is not None:
        _validate_equal_length("volatility", volatility, "point_ids", point_ids)
        data["point_id"] = list(point_ids)
    else:
        data["point_id"] = [f"p{i}" for i in range(n)]
    return pl.DataFrame(data)


def to_polars_backtest_frame(
    timestamps: Sequence[str],
    equity_curve: Sequence[float],
    returns: Sequence[float] | None = None,
    positions: Sequence[float] | None = None,
) -> pl.DataFrame:
    _validate_equal_length("timestamps", timestamps, "equity_curve", equity_curve)
    data: dict[str, object] = {"ts": list(timestamps), "equity": list(equity_curve)}
    if returns is not None:
        _validate_equal_length("timestamps", timestamps, "returns", returns)
        data["returns"] = list(returns)
    if positions is not None:
        _validate_equal_length("timestamps", timestamps, "positions", positions)
        data["position"] = list(positions)
    return pl.DataFrame(data).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))


@dataclass
class SignalStreamBuffer:
    """Incremental buffer for streaming signal updates in research notebooks."""

    _frames: list[pl.DataFrame]

    def __init__(self) -> None:
        self._frames = []

    def append(
        self,
        timestamps: Sequence[str],
        signal: Sequence[float],
        side: Sequence[float] | None = None,
        symbol: str | None = None,
    ) -> None:
        self._frames.append(to_polars_signal_frame(timestamps, signal, side=side, symbol=symbol))

    def frame(self) -> pl.DataFrame:
        if not self._frames:
            return pl.DataFrame({"ts": [], "signal": []})
        return pl.concat(self._frames, how="vertical")

    def clear(self) -> None:
        self._frames.clear()


def to_pandas(df: pl.DataFrame):  # type: ignore[no-untyped-def]
    """Optional pandas conversion for downstream tooling."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is not installed; install it to use to_pandas(), e.g. `uv add pandas`."
        ) from exc
    return df.to_pandas()
