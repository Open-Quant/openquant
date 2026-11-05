"""Source and symbol names must not escape or collapse cache namespaces."""

from __future__ import annotations

import polars as pl
import pytest
from openquant import data


def _source(name: str, price: float) -> data.CallableSource:
    def bars(symbol, start, end):
        return pl.DataFrame(
            {
                "date": ["2024-01-02"],
                "symbol": [symbol],
                "open": [price],
                "high": [price],
                "low": [price],
                "close": [price],
                "volume": [1.0],
            }
        )

    return data.CallableSource(bars, name=name)


@pytest.mark.parametrize(
    ("source_name", "symbol"),
    [(".", "X"), ("..", "X"), ("vendor", "."), ("vendor", ".."), ("..", "..")],
)
def test_dot_segments_keep_source_and_symbol_cache_directories(tmp_path, source_name, symbol):
    # Two levels inside tmp_path keep the old-code reproduction entirely inside this fixture.
    cache = tmp_path / "sandbox" / "cache"
    source = _source(source_name, 10.0)
    first = data.fetch(symbol, "2024-01-02", "2024-01-02", source=source, cache_dir=cache)
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    assert files[0].is_relative_to(cache)
    assert len(files[0].relative_to(cache).parts) == 3  # source / symbol / range
    cached = data.fetch(
        symbol, "2024-01-02", "2024-01-02", source=source, cache_dir=cache, offline=True
    )
    assert cached.equals(first)


def test_parent_directory_symbol_never_reuses_another_sources_prices(tmp_path):
    first_source, second_source = _source("first", 10.0), _source("second", 20.0)
    first = data.fetch("..", "2024-01-02", "2024-01-02", source=first_source, cache_dir=tmp_path)
    data.fetch("..", "2024-01-02", "2024-01-02", source=second_source, cache_dir=tmp_path)
    cached = data.fetch(
        "..", "2024-01-02", "2024-01-02", source=first_source, cache_dir=tmp_path, offline=True
    )
    assert first["close"].to_list() == [10.0]
    assert cached["close"].to_list() == [10.0]


def test_normal_dot_names_keep_existing_cache_layout(tmp_path):
    source = _source("vendor.example", 10.0)
    data.fetch("BRK.B", "2024-01-02", "2024-01-02", source=source, cache_dir=tmp_path)
    assert (tmp_path / "vendor.example" / "BRK.B" / "2024-01-02_2024-01-02.parquet").exists()
