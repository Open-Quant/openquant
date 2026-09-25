import pytest

from _core_fixtures import load_csv_columns

from openquant import fast_ewma


def test_ewma_matches_rust_reference():
    # Mirrors crates/openquant/tests/fast_ewma.rs::test_ewma
    (prices,) = load_csv_columns("microstructural_features/tick_data.csv", ["Price"])
    out = fast_ewma.ewma(prices, 20)

    assert len(out) == len(prices)
    assert out[0] == prices[0]
    # By hand: prices 1205, 1005 and alpha = 2/21 give
    # (1005 + (19/21) * 1205) / (1 + 19/21) = (21 * 1005 + 19 * 1205) / 40 = 1100.
    assert prices[:2] == [1205.0, 1005.0]
    assert abs(out[1] - 1100.0) < 1e-9


def test_ewma_matches_adjusted_pandas_recursion():
    # pandas `ewm(span=window, adjust=True).mean()`: weighted mean with weights (1-alpha)^k.
    values = [1.0, 2.0, 4.0, 8.0]
    window = 3
    decay = 1.0 - 2.0 / (window + 1)
    expected = []
    for end in range(len(values)):
        weights = [decay ** (end - i) for i in range(end + 1)]
        expected.append(sum(w * v for w, v in zip(weights, values)) / sum(weights))

    assert fast_ewma.ewma(values, window) == pytest.approx(expected, abs=1e-12)


def test_ewma_rejects_negative_window():
    with pytest.raises(OverflowError):
        fast_ewma.ewma([1.0, 2.0, 3.0], -1)


def test_ewma_zero_window_raises_value_error():
    with pytest.raises(ValueError):
        fast_ewma.ewma([1.0, 2.0, 3.0], 0)
