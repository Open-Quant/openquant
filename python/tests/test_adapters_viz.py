import polars as pl
import pytest

import openquant


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
    df = openquant.adapters.to_polars_indicator_matrix(ind, bar_index=[10, 11, 12], label_names=["a", "b", "c"])
    assert df.columns == ["bar_index", "a", "b", "c"]
    assert df["a"].to_list() == [1, 0, 1]


def test_to_polars_weights_frontier_backtest():
    w = openquant.adapters.to_polars_weights_frame(["A", "B"], [0.4, 0.6], as_of="2024-01-01 16:00:00")
    assert w.columns == ["asset", "weight", "as_of"]
    assert w["weight"].sum() == pytest.approx(1.0)

    frontier = openquant.adapters.to_polars_frontier_frame([0.1, 0.2], [0.08, 0.12], sharpe=[0.8, 0.6])
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
    fi = openquant.viz.prepare_feature_importance_payload(["f1", "f2", "f3"], [0.2, 0.7, 0.1], std=[0.01, 0.02, 0.03], top_n=2)
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
