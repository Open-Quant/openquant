from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from .data import _parse_ts


def prepare_feature_importance_payload(
    feature_names: Sequence[str],
    importance: Sequence[float],
    std: Sequence[float] | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Build a plotting-library-agnostic bar-chart payload for feature importances.

    Features are sorted by importance, highest first, before `top_n` is applied.

    Parameters
    ----------
    feature_names : Sequence[str]
        Feature names.
    importance : Sequence[float]
        Importance per feature.
    std : Sequence[float] or None, default None
        Optional error-bar size per feature.
    top_n : int or None, default None
        Keep only the `top_n` most important features.

    Returns
    -------
    dict[str, Any]
        `chart` (`"bar"`), `x` (feature names), `y` (importances), `y_label`
        (`"importance"`) and, if `std` is given, `error_y`.

    Raises
    ------
    ValueError
        If `importance` or `std` differs in length from `feature_names`.
    """
    if len(feature_names) != len(importance):
        raise ValueError("feature_names/importance length mismatch")
    if std is not None and len(std) != len(importance):
        raise ValueError("std/importance length mismatch")

    df = pl.DataFrame({"feature": list(feature_names), "importance": list(importance)})
    if std is not None:
        df = df.with_columns(pl.Series("std", list(std)))
    df = df.sort("importance", descending=True)
    if top_n is not None:
        df = df.head(top_n)

    payload: dict[str, Any] = {
        "chart": "bar",
        "x": df["feature"].to_list(),
        "y": df["importance"].to_list(),
        "y_label": "importance",
    }
    if "std" in df.columns:
        payload["error_y"] = df["std"].to_list()
    return payload


def prepare_feature_importance_comparison_payload(
    left_labels: Sequence[str],
    left_values: Sequence[float],
    right_labels: Sequence[str],
    right_values: Sequence[float],
    left_name: str = "left",
    right_name: str = "right",
) -> dict[str, Any]:
    """Build a grouped-bar payload comparing two sets of importances.

    The two sides are passed through as given (not sorted or aligned by label), so the
    label lists may differ, for example raw features against principal components.

    Parameters
    ----------
    left_labels : Sequence[str]
        Labels of the left series.
    left_values : Sequence[float]
        Values of the left series.
    right_labels : Sequence[str]
        Labels of the right series.
    right_values : Sequence[float]
        Values of the right series.
    left_name : str, default "left"
        Display name of the left series.
    right_name : str, default "right"
        Display name of the right series.

    Returns
    -------
    dict[str, Any]
        `chart` (`"grouped_bar"`), and `left` and `right`, each a dict with `name`, `labels`
        and `values`.

    Raises
    ------
    ValueError
        If a side's labels and values differ in length.
    """
    if len(left_labels) != len(left_values):
        raise ValueError("left_labels/left_values length mismatch")
    if len(right_labels) != len(right_values):
        raise ValueError("right_labels/right_values length mismatch")
    return {
        "chart": "grouped_bar",
        "left": {"name": left_name, "labels": list(left_labels), "values": list(left_values)},
        "right": {"name": right_name, "labels": list(right_labels), "values": list(right_values)},
    }


def prepare_drawdown_payload(
    timestamps: Sequence[str], equity_curve: Sequence[float]
) -> dict[str, Any]:
    """Build a line-chart payload of an equity curve and its drawdown.

    Drawdown is `equity / running_max(equity) - 1`, so it is 0 at each new high and
    negative below it; it assumes a positive equity curve. Timestamps are parsed and
    re-rendered with `str`, so a timestamp that fails to parse comes out as `"None"`.

    Parameters
    ----------
    timestamps : Sequence[str]
        Timestamp strings, one per equity value.
    equity_curve : Sequence[float]
        Equity values.

    Returns
    -------
    dict[str, Any]
        `chart` (`"line"`), `x` (timestamps as strings), `equity`, `drawdown` and `y_label`
        (`"drawdown"`).

    Raises
    ------
    ValueError
        If `equity_curve` differs in length from `timestamps`.
    """
    if len(timestamps) != len(equity_curve):
        raise ValueError("timestamps/equity_curve length mismatch")
    df = pl.DataFrame({"ts": list(timestamps), "equity": list(equity_curve)}).with_columns(
        _parse_ts(pl.col("ts"))
    )
    df = df.with_columns((pl.col("equity") / pl.col("equity").cum_max() - 1.0).alias("drawdown"))
    return {
        "chart": "line",
        "x": [str(x) for x in df["ts"].to_list()],
        "equity": df["equity"].to_list(),
        "drawdown": df["drawdown"].to_list(),
        "y_label": "drawdown",
    }


def prepare_regime_payload(
    timestamps: Sequence[str],
    score: Sequence[float],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Build a line-plus-step payload of a regime score and the regime it implies.

    Each point is regime 1 when `score >= threshold` and -1 otherwise. Timestamps are
    passed through unparsed.

    Parameters
    ----------
    timestamps : Sequence[str]
        Timestamp per score.
    score : Sequence[float]
        Regime score.
    threshold : float, default 0.0
        Score at or above which the regime is 1.

    Returns
    -------
    dict[str, Any]
        `chart` (`"line+step"`), `x` (timestamps), `score`, `threshold` and `regime`
        (list of 1/-1).

    Raises
    ------
    ValueError
        If `score` differs in length from `timestamps`.
    """
    if len(timestamps) != len(score):
        raise ValueError("timestamps/score length mismatch")
    regimes = [1 if s >= threshold else -1 for s in score]
    return {
        "chart": "line+step",
        "x": list(timestamps),
        "score": list(score),
        "threshold": threshold,
        "regime": regimes,
    }


def prepare_frontier_payload(
    volatility: Sequence[float],
    returns: Sequence[float],
    sharpe: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a scatter payload of an efficient frontier, optionally colored by Sharpe ratio.

    Parameters
    ----------
    volatility : Sequence[float]
        Volatility of each point (x axis).
    returns : Sequence[float]
        Return of each point (y axis).
    sharpe : Sequence[float] or None, default None
        Optional Sharpe ratio per point, used as the color.

    Returns
    -------
    dict[str, Any]
        `chart` (`"scatter"`), `x`, `y`, `x_label` (`"volatility"`), `y_label`
        (`"return"`) and, if `sharpe` is given, `color` and `color_label` (`"sharpe"`).

    Raises
    ------
    ValueError
        If `returns` or `sharpe` differs in length from `volatility`.
    """
    if len(volatility) != len(returns):
        raise ValueError("volatility/returns length mismatch")
    payload: dict[str, Any] = {
        "chart": "scatter",
        "x": list(volatility),
        "y": list(returns),
        "x_label": "volatility",
        "y_label": "return",
    }
    if sharpe is not None:
        if len(sharpe) != len(volatility):
            raise ValueError("sharpe/volatility length mismatch")
        payload["color"] = list(sharpe)
        payload["color_label"] = "sharpe"
    return payload


def prepare_cluster_payload(
    node_id: Sequence[str],
    parent_id: Sequence[str | None],
    height: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a tree payload (for example a dendrogram) from parent links.

    Parameters
    ----------
    node_id : Sequence[str]
        Identifier of each node.
    parent_id : Sequence[str or None]
        Parent of each node, None for a root. Not checked against `node_id`.
    height : Sequence[float] or None, default None
        Optional height (for example merge distance) per node.

    Returns
    -------
    dict[str, Any]
        `chart` (`"tree"`), `node_id`, `parent_id` and, if given, `height`.

    Raises
    ------
    ValueError
        If `parent_id` or `height` differs in length from `node_id`.
    """
    if len(node_id) != len(parent_id):
        raise ValueError("node_id/parent_id length mismatch")
    if height is not None and len(height) != len(node_id):
        raise ValueError("height/node_id length mismatch")
    payload: dict[str, Any] = {
        "chart": "tree",
        "node_id": list(node_id),
        "parent_id": list(parent_id),
    }
    if height is not None:
        payload["height"] = list(height)
    return payload
