from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from .data import _parse_ts


def _validate_equal_length(
    name_a: str, a: Sequence[object], name_b: str, b: Sequence[object]
) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a}/{name_b} length mismatch: {len(a)} vs {len(b)}")


def to_polars_signal_frame(
    timestamps: Sequence[str],
    signal: Sequence[float],
    side: Sequence[float] | None = None,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Build a signal frame from parallel timestamp and signal lists.

    `ts` strings are parsed to `Datetime`: `"%Y-%m-%d %H:%M:%S"` with an optional fractional
    second is parsed exactly, anything else falls back to polars' format inference, and a
    string that parses under neither becomes null (no error is raised).

    Parameters
    ----------
    timestamps : Sequence[str]
        Timestamp strings, one per signal value.
    signal : Sequence[float]
        Signal values.
    side : Sequence[float] or None, default None
        Optional side (for example +1/-1) per row.
    symbol : str or None, default None
        Optional symbol, repeated on every row.

    Returns
    -------
    polars.DataFrame
        Columns `ts` (Datetime) and `signal`, then `side` if given and `symbol` if given,
        in input order.

    Raises
    ------
    ValueError
        If `signal` or `side` differs in length from `timestamps`.
    """
    _validate_equal_length("timestamps", timestamps, "signal", signal)
    data: dict[str, Any] = {"ts": list(timestamps), "signal": list(signal)}
    if side is not None:
        _validate_equal_length("timestamps", timestamps, "side", side)
        data["side"] = list(side)
    if symbol is not None:
        data["symbol"] = [symbol] * len(timestamps)
    return pl.DataFrame(data).with_columns(_parse_ts(pl.col("ts")))


def to_polars_event_frame(
    starts: Sequence[str],
    ends: Sequence[str],
    probs: Sequence[float],
    sides: Sequence[float] | None = None,
    labels: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Build an event frame from parallel start/end timestamp and probability lists.

    `start_ts` and `end_ts` strings are parsed to `Datetime` the same way as in
    `to_polars_signal_frame`; unparseable strings become null.

    Parameters
    ----------
    starts : Sequence[str]
        Event start timestamps.
    ends : Sequence[str]
        Event end timestamps.
    probs : Sequence[float]
        Predicted probability per event.
    sides : Sequence[float] or None, default None
        Optional side per event.
    labels : Sequence[int] or None, default None
        Optional label per event.

    Returns
    -------
    polars.DataFrame
        Columns `start_ts` and `end_ts` (Datetime) and `prob`, then `side` and `label` if
        given, in input order.

    Raises
    ------
    ValueError
        If `ends`, `probs`, `sides` or `labels` differs in length from `starts`.
    """
    _validate_equal_length("starts", starts, "ends", ends)
    _validate_equal_length("starts", starts, "probs", probs)
    data: dict[str, Any] = {"start_ts": list(starts), "end_ts": list(ends), "prob": list(probs)}
    if sides is not None:
        _validate_equal_length("starts", starts, "sides", sides)
        data["side"] = list(sides)
    if labels is not None:
        _validate_equal_length("starts", starts, "labels", labels)
        data["label"] = list(labels)
    return pl.DataFrame(data).with_columns(
        _parse_ts(pl.col("start_ts")),
        _parse_ts(pl.col("end_ts")),
    )


def to_polars_indicator_matrix(
    ind_mat: Sequence[Sequence[int]],
    bar_index: Sequence[int] | None = None,
    label_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Convert a row-major indicator matrix into a wide frame, one column per label.

    Typically the bar-by-label indicator matrix of AFML section 4.5 (Snippet 4.3), where
    entry `[t][i]` is 1 when bar `t` falls inside label `i`'s span. Values are cast to `int`.

    Parameters
    ----------
    ind_mat : Sequence[Sequence[int]]
        Rectangular matrix, one row per bar and one column per label.
    bar_index : Sequence[int] or None, default None
        Index for each row; defaults to `0..len(ind_mat) - 1`.
    label_names : Sequence[str] or None, default None
        Column names for the labels; defaults to `label_0`, `label_1`, ...

    Returns
    -------
    polars.DataFrame
        Column `bar_index` followed by one integer column per label. An empty `ind_mat`
        gives a frame with only an empty `bar_index` column.

    Raises
    ------
    ValueError
        If `ind_mat` is not rectangular, or `label_names` or `bar_index` has the wrong
        length.
    """
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

    data: dict[str, Any] = {"bar_index": list(bar_index)}
    for j, name in enumerate(label_names):
        data[name] = [int(row[j]) for row in ind_mat]
    return pl.DataFrame(data)


def to_polars_weights_frame(
    asset_names: Sequence[str],
    weights: Sequence[float],
    as_of: str | None = None,
) -> pl.DataFrame:
    """Build a portfolio weights frame from parallel asset name and weight lists.

    Parameters
    ----------
    asset_names : Sequence[str]
        Asset names.
    weights : Sequence[float]
        Weight per asset.
    as_of : str or None, default None
        Optional timestamp string, repeated on every row and parsed to `Datetime`
        (null if it does not parse).

    Returns
    -------
    polars.DataFrame
        Columns `asset` and `weight`, plus `as_of` (Datetime) if given.

    Raises
    ------
    ValueError
        If `weights` differs in length from `asset_names`.
    """
    _validate_equal_length("asset_names", asset_names, "weights", weights)
    data: dict[str, Any] = {"asset": list(asset_names), "weight": list(weights)}
    if as_of is not None:
        data["as_of"] = [as_of] * len(asset_names)
    df = pl.DataFrame(data)
    if as_of is not None:
        df = df.with_columns(_parse_ts(pl.col("as_of")))
    return df


def to_polars_frontier_frame(
    volatility: Sequence[float],
    returns: Sequence[float],
    sharpe: Sequence[float] | None = None,
    point_ids: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build an efficient-frontier frame from parallel volatility and return lists.

    Parameters
    ----------
    volatility : Sequence[float]
        Volatility of each frontier point.
    returns : Sequence[float]
        Expected return of each frontier point.
    sharpe : Sequence[float] or None, default None
        Optional Sharpe ratio per point.
    point_ids : Sequence[str] or None, default None
        Optional identifier per point; defaults to `p0`, `p1`, ...

    Returns
    -------
    polars.DataFrame
        Columns `volatility`, `return`, `sharpe` (only if given) and `point_id`.

    Raises
    ------
    ValueError
        If `returns`, `sharpe` or `point_ids` differs in length from `volatility`.
    """
    _validate_equal_length("volatility", volatility, "returns", returns)
    n = len(volatility)
    data: dict[str, Any] = {"volatility": list(volatility), "return": list(returns)}
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
    """Build a backtest frame from parallel timestamp and equity-curve lists.

    `ts` strings are parsed to `Datetime` the same way as in `to_polars_signal_frame`;
    unparseable strings become null.

    Parameters
    ----------
    timestamps : Sequence[str]
        Timestamp strings, one per equity value.
    equity_curve : Sequence[float]
        Equity value per timestamp.
    returns : Sequence[float] or None, default None
        Optional per-bar strategy return.
    positions : Sequence[float] or None, default None
        Optional position per bar.

    Returns
    -------
    polars.DataFrame
        Columns `ts` (Datetime) and `equity`, then `returns` and `position` if given.

    Raises
    ------
    ValueError
        If `equity_curve`, `returns` or `positions` differs in length from `timestamps`.
    """
    _validate_equal_length("timestamps", timestamps, "equity_curve", equity_curve)
    data: dict[str, Any] = {"ts": list(timestamps), "equity": list(equity_curve)}
    if returns is not None:
        _validate_equal_length("timestamps", timestamps, "returns", returns)
        data["returns"] = list(returns)
    if positions is not None:
        _validate_equal_length("timestamps", timestamps, "positions", positions)
        data["position"] = list(positions)
    return pl.DataFrame(data).with_columns(_parse_ts(pl.col("ts")))


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
        """Append one batch of signal rows.

        The batch is converted with `to_polars_signal_frame` immediately, so length errors
        surface here. Use the same optional columns (`side`, `symbol`) in every batch: `frame`
        concatenates batches vertically and fails if their columns differ.

        Parameters
        ----------
        timestamps : Sequence[str]
            Timestamp strings, one per signal value.
        signal : Sequence[float]
            Signal values.
        side : Sequence[float] or None, default None
            Optional side per row.
        symbol : str or None, default None
            Optional symbol, repeated on every row of the batch.

        Raises
        ------
        ValueError
            If `signal` or `side` differs in length from `timestamps`.
        """
        self._frames.append(to_polars_signal_frame(timestamps, signal, side=side, symbol=symbol))

    def frame(self) -> pl.DataFrame:
        """Return all buffered batches as one frame, in append order.

        Returns
        -------
        polars.DataFrame
            The vertical concatenation of every appended batch (columns as in
            `to_polars_signal_frame`). With nothing buffered, an empty frame with columns `ts`
            and `signal`.
        """
        if not self._frames:
            return pl.DataFrame({"ts": [], "signal": []})
        return pl.concat(self._frames, how="vertical")

    def clear(self) -> None:
        """Discard all buffered batches."""
        self._frames.clear()


def to_pandas(df: pl.DataFrame) -> Any:
    """Optional pandas conversion for downstream tooling."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is not installed; install it to use to_pandas(), e.g. `uv add pandas`."
        ) from exc
    return df.to_pandas()
