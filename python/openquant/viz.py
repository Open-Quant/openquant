from __future__ import annotations

from typing import Sequence

import polars as pl


def prepare_feature_importance_payload(
    feature_names: Sequence[str],
    importance: Sequence[float],
    std: Sequence[float] | None = None,
    top_n: int | None = None,
) -> dict[str, object]:
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

    payload: dict[str, object] = {
        "chart": "bar",
        "x": df["feature"].to_list(),
        "y": df["importance"].to_list(),
        "y_label": "importance",
    }
    if "std" in df.columns:
        payload["error_y"] = df["std"].to_list()
    return payload


def prepare_drawdown_payload(timestamps: Sequence[str], equity_curve: Sequence[float]) -> dict[str, object]:
    if len(timestamps) != len(equity_curve):
        raise ValueError("timestamps/equity_curve length mismatch")
    df = pl.DataFrame({"ts": list(timestamps), "equity": list(equity_curve)}).with_columns(
        pl.col("ts").str.strptime(pl.Datetime, strict=False)
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
) -> dict[str, object]:
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
) -> dict[str, object]:
    if len(volatility) != len(returns):
        raise ValueError("volatility/returns length mismatch")
    payload: dict[str, object] = {
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
) -> dict[str, object]:
    if len(node_id) != len(parent_id):
        raise ValueError("node_id/parent_id length mismatch")
    if height is not None and len(height) != len(node_id):
        raise ValueError("height/node_id length mismatch")
    payload: dict[str, object] = {"chart": "tree", "node_id": list(node_id), "parent_id": list(parent_id)}
    if height is not None:
        payload["height"] = list(height)
    return payload
