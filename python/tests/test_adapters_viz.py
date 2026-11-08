import openquant
import pytest


def test_to_polars_signal_frame_and_stream_buffer():
    df = openquant.adapters.to_polars_signal_frame(
        ["2024-01-01 09:30:00", "2024-01-01 09:31:00"], [0.1, -0.2], side=[1.0, -1.0], symbol="ES"
    )
    assert df.columns == ["ts", "signal", "side", "symbol"]
    assert df.height == 2
    assert df["symbol"].to_list() == ["ES", "ES"]

    buf = openquant.adapters.SignalStreamBuffer()
    buf.append(["2024-01-01 09:30:00"], [0.1], symbol="NQ")
    buf.append(["2024-01-01 09:31:00"], [0.2], symbol="NQ")
    out = buf.frame()
    assert out.height == 2
    assert out["symbol"].to_list() == ["NQ", "NQ"]
    buf.clear()
    assert buf.frame().height == 0


def test_to_polars_event_frame():
    df = openquant.adapters.to_polars_event_frame(
        starts=["2024-01-01 09:30:00", "2024-01-01 09:35:00"],
        ends=["2024-01-01 10:00:00", "2024-01-01 10:05:00"],
        probs=[0.6, 0.55],
        sides=[1.0, -1.0],
        labels=[1, 0],
    )
    assert df.columns == ["start_ts", "end_ts", "prob", "side", "label"]
    assert df.height == 2


def test_to_polars_indicator_matrix():
    ind = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    df = openquant.adapters.to_polars_indicator_matrix(
        ind, bar_index=[10, 11, 12], label_names=["a", "b", "c"]
    )
    assert df.columns == ["bar_index", "a", "b", "c"]
    assert df["a"].to_list() == [1, 0, 1]


def test_to_polars_weights_frontier_backtest():
    w = openquant.adapters.to_polars_weights_frame(
        ["A", "B"], [0.4, 0.6], as_of="2024-01-01 16:00:00"
    )
    assert w.columns == ["asset", "weight", "as_of"]
    assert w["weight"].sum() == pytest.approx(1.0)

    frontier = openquant.adapters.to_polars_frontier_frame(
        [0.1, 0.2], [0.08, 0.12], sharpe=[0.8, 0.6]
    )
    assert frontier.columns == ["volatility", "return", "sharpe", "point_id"]
    assert frontier.height == 2

    bt = openquant.adapters.to_polars_backtest_frame(
        ["2024-01-01 09:30:00", "2024-01-01 09:31:00"],
        [100.0, 100.5],
        returns=[0.0, 0.005],
        positions=[0.0, 1.0],
    )
    assert bt.columns == ["ts", "equity", "returns", "position"]
    assert bt.height == 2


def test_viz_payloads():
    fi = openquant.viz.prepare_feature_importance_payload(
        ["f1", "f2", "f3"], [0.2, 0.7, 0.1], std=[0.01, 0.02, 0.03], top_n=2
    )
    assert fi["chart"] == "bar"
    assert fi["x"] == ["f2", "f1"]

    dd = openquant.viz.prepare_drawdown_payload(
        ["2024-01-01 09:30:00", "2024-01-01 09:31:00", "2024-01-01 09:32:00"],
        [100.0, 99.0, 101.0],
    )
    assert dd["chart"] == "line"
    assert len(dd["drawdown"]) == 3

    reg = openquant.viz.prepare_regime_payload(["t1", "t2", "t3"], [0.3, -0.2, 0.1], threshold=0.0)
    assert reg["regime"] == [1, -1, 1]

    frontier = openquant.viz.prepare_frontier_payload([0.1, 0.2], [0.08, 0.12], sharpe=[0.8, 0.6])
    assert frontier["chart"] == "scatter"
    assert frontier["color"] == [0.8, 0.6]

    comparison = openquant.viz.prepare_feature_importance_comparison_payload(
        ["f1", "f2"], [0.6, 0.4], ["pc1", "pc2"], [0.55, 0.45]
    )
    assert comparison["chart"] == "grouped_bar"

    cluster = openquant.viz.prepare_cluster_payload(["n1", "n2"], [None, "n1"], height=[0.0, 1.2])
    assert cluster["chart"] == "tree"
    assert cluster["height"] == [0.0, 1.2]


def test_adapter_shape_validation():
    with pytest.raises(ValueError):
        openquant.adapters.to_polars_signal_frame(["2024-01-01 09:30:00"], [0.1, 0.2])
    with pytest.raises(ValueError):
        openquant.adapters.to_polars_indicator_matrix([[1, 0], [1]], bar_index=[0, 1])

    with pytest.raises(ValueError):
        openquant.viz.prepare_frontier_payload([0.1], [0.1, 0.2])


def test_to_pandas_bridge():
    df = openquant.adapters.to_polars_weights_frame(["A"], [1.0])
    try:
        pdf = openquant.adapters.to_pandas(df)
        assert list(pdf.columns) == ["asset", "weight"]
    except RuntimeError as err:
        assert "pandas is not installed" in str(err)


@pytest.mark.parametrize("names", [["a", "a"], ["bar_index", "a"]])
def test_indicator_matrix_rejects_column_name_collisions(names):
    with pytest.raises(ValueError, match="label_names"):
        openquant.adapters.to_polars_indicator_matrix(
            [[1, 0], [0, 1]], bar_index=[10, 11], label_names=names
        )


def test_indicator_matrix_accepts_numpy_rows():
    import numpy as np

    result = openquant.adapters.to_polars_indicator_matrix(np.array([[1, 0], [0, 1]]))
    assert result.to_dict(as_series=False) == {
        "bar_index": [0, 1],
        "label_0": [1, 0],
        "label_1": [0, 1],
    }


@pytest.mark.parametrize(
    "factory, kwargs, timestamp_columns",
    [
        ("to_polars_signal_frame", {"timestamps": [], "signal": []}, ["ts"]),
        (
            "to_polars_event_frame",
            {"starts": [], "ends": [], "probs": [], "sides": [], "labels": []},
            ["start_ts", "end_ts"],
        ),
        (
            "to_polars_weights_frame",
            {"asset_names": [], "weights": [], "as_of": "2024-01-01"},
            ["as_of"],
        ),
        (
            "to_polars_backtest_frame",
            {"timestamps": [], "equity_curve": [], "returns": [], "positions": []},
            ["ts"],
        ),
    ],
)
def test_empty_timestamped_adapter_frames(factory, kwargs, timestamp_columns):
    import polars as pl

    result = getattr(openquant.adapters, factory)(**kwargs)
    assert result.height == 0
    for column in timestamp_columns:
        assert result.schema[column] == pl.Datetime("us")


def test_signal_buffer_normalizes_numeric_batch_types_and_empty_batches():
    import polars as pl

    buffer = openquant.adapters.SignalStreamBuffer()
    buffer.append([], [], side=[], symbol="CL")
    buffer.append(["2024-01-01 09:30:00"], [1], side=[-1], symbol="CL")
    buffer.append(["2024-01-01 09:31:00"], [0.5], side=[1.0], symbol="CL")
    buffer.append(["2024-01-01 09:32:00"], [None], side=[None], symbol="CL")
    buffer.append([], [], side=[], symbol="CL")
    result = buffer.frame()
    assert result["signal"].to_list() == [1.0, 0.5, None]
    assert result["side"].to_list() == [-1.0, 1.0, None]
    assert result["symbol"].to_list() == ["CL"] * 3
    assert result.schema["signal"] == pl.Float64
    assert result.schema["side"] == pl.Float64


def test_signal_buffer_preserves_optional_column_mismatch_error():
    import polars as pl

    buffer = openquant.adapters.SignalStreamBuffer()
    buffer.append(["2024-01-01 09:30:00"], [1.0])
    buffer.append(["2024-01-01 09:31:00"], [0.5], symbol="CL")
    with pytest.raises(pl.exceptions.ShapeError):
        buffer.frame()


def test_empty_and_cleared_signal_buffers_have_signal_schema():
    import polars as pl

    buffer = openquant.adapters.SignalStreamBuffer()
    expected = {"ts": pl.Datetime("us"), "signal": pl.Float64}
    assert buffer.frame().schema == expected
    buffer.append(["2024-01-01 09:30:00"], [1.0], symbol="CL")
    buffer.clear()
    assert buffer.frame().schema == expected


def test_signal_frame_accepts_mixed_numeric_values_in_one_batch():
    import polars as pl

    result = openquant.adapters.to_polars_signal_frame(
        ["2024-01-01 09:30:00", "2024-01-01 09:31:00"], [1, 0.5], side=[1, -0.5]
    )
    assert result["signal"].to_list() == [1.0, 0.5]
    assert result["side"].to_list() == [1.0, -0.5]
    assert result.schema["signal"] == pl.Float64


def test_signal_buffer_accepts_null_only_first_batch():
    buffer = openquant.adapters.SignalStreamBuffer()
    buffer.append(["2024-01-01 09:30:00"], [None])
    buffer.append(["2024-01-01 09:31:00"], [0.5])
    assert buffer.frame()["signal"].to_list() == [None, 0.5]
