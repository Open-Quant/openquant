"""Finite thresholds are required even if cleaning leaves no trades (#278)."""

import math

import openquant
import polars as pl
import pytest

BUILDERS = [("build_volume_bars", "volume_per_bar"), ("build_dollar_bars", "dollar_value_per_bar")]


def _frame():
    return pl.DataFrame(
        {
            "ts": ["2024-01-01 09:30:00"],
            "symbol": ["SYNTHETIC"],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [2.0],
        }
    )


@pytest.mark.parametrize("builder,parameter", BUILDERS)
@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, 0.0, -1.0])
@pytest.mark.parametrize("input_kind", ["populated", "empty", "cleaned_empty"])
def test_wrapper_rejects_invalid_threshold_before_grouping(
    builder, parameter, threshold, input_kind
):
    frame = _frame()
    if input_kind == "empty":
        frame = frame.head(0)
    elif input_kind == "cleaned_empty":
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("close"))
    with pytest.raises(ValueError, match=rf"{parameter}.*finite.*> 0"):
        getattr(openquant.bars, builder)(frame, **{parameter: threshold})


@pytest.mark.parametrize(
    "builder,parameter,kwargs",
    [
        ("build_volume_bars", "volume_per_bar", {}),
        ("build_dollar_bars", "dollar_value_per_bar", {}),
        ("build_imbalance_bars", "threshold", {"bar_type": "tick"}),
    ],
)
@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_native_threshold_error_names_finiteness(builder, parameter, kwargs, threshold):
    with pytest.raises(ValueError, match=rf"{parameter}.*finite.*> 0"):
        getattr(openquant._core.bars, builder)([], [], [], **{parameter: threshold}, **kwargs)


@pytest.mark.parametrize("builder,parameter", BUILDERS)
def test_positive_subnormal_threshold_still_closes_one_trade_bar(builder, parameter):
    # A single trade's volume (2) and dollar value (20) exceed the smallest
    # positive binary64 threshold; the public builder must retain that trade.
    result = getattr(openquant.bars, builder)(_frame(), **{parameter: math.ulp(0.0)})
    assert result.height == 1
    assert result["n_obs"].to_list() == [1]
    assert result["volume"].to_list() == [2.0]
    assert result["dollar_value"].to_list() == [20.0]


@pytest.mark.parametrize("builder,parameter", BUILDERS)
def test_positive_threshold_with_empty_input_remains_empty(builder, parameter):
    result = getattr(openquant.bars, builder)(_frame().head(0), **{parameter: 1.0})
    assert result.height == 0
    assert "n_obs" in result.columns
