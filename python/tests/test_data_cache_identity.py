"""A valid content hash alone cannot establish which request produced a cache entry."""

from __future__ import annotations

import json

import pytest
from openquant import data


def _source(price: float) -> data.CallableSource:
    def bars(symbol, start, end):
        return {
            "date": ["2024-01-02"],
            "open": [price],
            "high": [price],
            "low": [price],
            "close": [price],
            "volume": [1.0],
        }

    return data.CallableSource(bars, name="vendor", version="v1")


def _populate(cache):
    data.fetch("AAA", "2024-01-02", "2024-01-02", source=_source(10.0), cache_dir=cache)
    sidecar = next(cache.rglob("*.json"))
    return sidecar, json.loads(sidecar.read_text())


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("format", "unknown-format"),
        ("source", "another-vendor"),
        ("source_version", "v2"),
        ("symbol", "BBB"),
        ("start", "2024-01-01"),
        ("end", "2024-01-03"),
    ],
)
@pytest.mark.parametrize("offline", [False, True])
def test_mismatched_request_metadata_is_not_a_cache_hit(tmp_path, field, wrong, offline):
    sidecar, metadata = _populate(tmp_path)
    metadata[field] = wrong  # Keep the valid data and its content hash unchanged.
    sidecar.write_text(json.dumps(metadata))
    source = _source(20.0)
    if offline:
        with pytest.raises(data.CacheMissError):
            data.fetch(
                "AAA", "2024-01-02", "2024-01-02", source=source, cache_dir=tmp_path, offline=True
            )
    else:
        frame, result = data.fetch(
            "AAA", "2024-01-02", "2024-01-02", source=source, cache_dir=tmp_path, return_meta=True
        )
        assert frame["close"].to_list() == [20.0]
        assert result["cache"] == {"AAA": "miss"}
        assert json.loads(sidecar.read_text())[field] != wrong


@pytest.mark.parametrize("contents", ["{", "null", "[]", '"not an object"', "{}", b"\xff"])
@pytest.mark.parametrize("offline", [False, True])
def test_malformed_sidecar_is_refetched_or_reported_as_cache_miss(tmp_path, contents, offline):
    sidecar, _ = _populate(tmp_path)
    if isinstance(contents, bytes):
        sidecar.write_bytes(contents)
    else:
        sidecar.write_text(contents)
    source = _source(20.0)
    if offline:
        with pytest.raises(data.CacheMissError):
            data.fetch(
                "AAA", "2024-01-02", "2024-01-02", source=source, cache_dir=tmp_path, offline=True
            )
    else:
        frame = data.fetch("AAA", "2024-01-02", "2024-01-02", source=source, cache_dir=tmp_path)
        assert frame["close"].to_list() == [20.0]
        assert json.loads(sidecar.read_text())["format"] == data.CACHE_FORMAT


def test_complete_matching_sidecar_is_still_an_offline_hit(tmp_path):
    _populate(tmp_path)
    frame, result = data.fetch(
        "AAA",
        "2024-01-02",
        "2024-01-02",
        source=_source(20.0),
        cache_dir=tmp_path,
        offline=True,
        return_meta=True,
    )
    assert frame["close"].to_list() == [10.0]
    assert result["cache"] == {"AAA": "hit"}
