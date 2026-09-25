"""`openquant.data.fetch`: sources, the on-disk cache, the content hash, manifests."""

from __future__ import annotations

from datetime import date
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

import openquant
from openquant import data

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SYMBOLS = ["SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_E"]
# Pinned: the sample is committed and the hash definition is versioned, so this
# value changes only if one of them does (and then both need a deliberate update).
FULL_SAMPLE_HASH = "sha256:b311c219efb2a19db449767cc8600139c030aad15496b011fc650b4880550eea"


def _bars(symbol: str, start: date, end: date, *, close0: float = 10.0) -> pl.DataFrame:
    days = pl.date_range(start, end, "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)
    n = len(days)
    close = [close0 + 0.1 * i for i in range(n)]
    return pl.DataFrame(
        {
            "Date": days.cast(pl.Utf8),
            "Open": close,
            "High": [c + 0.5 for c in close],
            "Low": [c - 0.5 for c in close],
            "Close": close,
            "Volume": [1000 + i for i in range(n)],
        }
    )


class CountingSource:
    """A user-style source (duck-typed, no base class) that counts its calls."""

    name = "test-counting"
    terms = "test data"

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []

    def fetch_symbol(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        self.calls.append((symbol, start, end))
        return _bars(symbol, start, end)


# --- fetch + cache -----------------------------------------------------------


def test_fetch_default_sample_passes_quality_report(tmp_path):
    df = data.fetch(["SYN_A", "SYN_B"], "2022-01-01", "2022-06-30", cache_dir=tmp_path)

    assert df.columns == data.CANONICAL_OHLCV_COLUMNS
    assert df["symbol"].unique().sort().to_list() == ["SYN_A", "SYN_B"]
    assert df["ts"].min().date() == date(2022, 1, 3)
    assert df["ts"].max().date() <= date(2022, 6, 30)
    report = data.data_quality_report(df)
    assert data.quality_failures(report) == []
    assert report["row_count"] == df.height > 0
    assert report["duplicate_key_count"] == 0
    # The canonical frame round-trips through clean_ohlcv unchanged.
    assert data.clean_ohlcv(df).equals(df)


def test_second_call_is_served_from_cache_without_the_source(tmp_path, monkeypatch):
    src = CountingSource()
    first, meta1 = data.fetch(["AAA", "BBB"], "2024-01-01", "2024-02-29", source=src, cache_dir=tmp_path, return_meta=True)
    assert len(src.calls) == 2
    assert meta1["cache"] == {"AAA": "miss", "BBB": "miss"}

    def offline(*_args, **_kwargs):
        raise ConnectionError("network is down")

    monkeypatch.setattr(src, "fetch_symbol", offline)
    second, meta2 = data.fetch(["AAA", "BBB"], "2024-01-01", "2024-02-29", source=src, cache_dir=tmp_path, return_meta=True)
    assert meta2["cache"] == {"AAA": "hit", "BBB": "hit"}
    assert second.equals(first)
    assert meta2["dataset_hash"] == meta1["dataset_hash"]

    # offline=True also succeeds, and a request that is not cached raises instead of fetching.
    assert data.fetch(["BBB", "AAA"], "2024-01-01", "2024-02-29", source=src, cache_dir=tmp_path, offline=True).equals(first)
    with pytest.raises(data.CacheMissError):
        data.fetch(["AAA"], "2024-01-01", "2024-03-29", source=src, cache_dir=tmp_path, offline=True)
    with pytest.raises(ConnectionError):
        data.fetch(["CCC"], "2024-01-01", "2024-02-29", source=src, cache_dir=tmp_path)


def test_cache_layout_and_sidecar(tmp_path):
    src = CountingSource()
    df = data.fetch("A/B", "2024-01-01", "2024-01-31", source=src, cache_dir=tmp_path)
    entry_dir = tmp_path / "test-counting" / "A%2FB"
    assert sorted(p.name for p in entry_dir.iterdir()) == ["2024-01-01_2024-01-31.json", "2024-01-01_2024-01-31.parquet"]
    sidecar = json.loads((entry_dir / "2024-01-01_2024-01-31.json").read_text())
    assert sidecar["format"] == data.CACHE_FORMAT
    assert sidecar["source"] == "test-counting"
    assert sidecar["symbol"] == "A/B"
    assert sidecar["terms"] == "test data"
    assert sidecar["rows"] == df.height
    assert sidecar["dataset_hash"] == data.dataset_hash(pl.read_parquet(entry_dir / "2024-01-01_2024-01-31.parquet"))


def test_version_and_refresh_and_tampering_refetch(tmp_path):
    src = CountingSource()
    data.fetch("AAA", "2024-01-01", "2024-01-31", source=src, cache_dir=tmp_path)
    data.fetch("AAA", "2024-01-01", "2024-01-31", source=src, cache_dir=tmp_path, refresh=True)
    assert len(src.calls) == 2

    src.version = "v2"  # a new version is a new cache namespace
    data.fetch("AAA", "2024-01-01", "2024-01-31", source=src, cache_dir=tmp_path)
    assert len(src.calls) == 3

    # A cache file edited behind fetch's back no longer matches its recorded hash.
    parquet = next((tmp_path / "test-counting@v2" / "AAA").glob("*.parquet"))
    pl.read_parquet(parquet).with_columns(pl.col("close") * 2).write_parquet(parquet)
    with pytest.raises(data.CacheMissError):
        data.fetch("AAA", "2024-01-01", "2024-01-31", source=src, cache_dir=tmp_path, offline=True)
    data.fetch("AAA", "2024-01-01", "2024-01-31", source=src, cache_dir=tmp_path)
    assert len(src.calls) == 4


def test_default_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENQUANT_DATA_CACHE", str(tmp_path / "c"))
    assert data.default_cache_dir() == tmp_path / "c"
    monkeypatch.delenv("OPENQUANT_DATA_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert data.default_cache_dir() == tmp_path / "xdg" / "openquant" / "data"
    data.fetch("SYN_C", "2023-01-01", "2023-01-31")
    assert any((tmp_path / "xdg" / "openquant" / "data").rglob("*.parquet"))


def test_fetch_rejects_bad_requests(tmp_path):
    with pytest.raises(ValueError, match="after end"):
        data.fetch("SYN_A", "2023-02-01", "2023-01-01", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="symbols"):
        data.fetch([], "2023-01-01", "2023-02-01", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="ISO date"):
        data.fetch("SYN_A", "01/02/2023", "2023-02-01", cache_dir=tmp_path)
    with pytest.raises(LookupError, match="not in"):
        data.fetch("AAPL", "2023-01-01", "2023-02-01", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="no usable rows"):
        data.fetch("SYN_A", "2030-01-01", "2030-02-01", cache_dir=tmp_path)
    with pytest.raises(TypeError, match="fetch_symbol"):
        data.fetch("SYN_A", "2023-01-01", "2023-02-01", source=object(), cache_dir=tmp_path)
    assert not any(tmp_path.rglob("*.parquet"))


def test_source_output_is_cleaned_and_checked(tmp_path):
    def messy(symbol, start, end):
        bars = _bars(symbol, date(2023, 12, 1), date(2024, 2, 15)).with_columns(pl.lit(symbol).alias("ticker"))
        return pl.concat([bars, bars.head(3)])  # duplicates and rows outside the range

    df = data.fetch("X", "2024-01-01", "2024-01-31", source=data.CallableSource(messy, name="messy"), cache_dir=tmp_path)
    assert df["ts"].dt.date().min() >= date(2024, 1, 1)
    assert df["ts"].dt.date().max() <= date(2024, 1, 31)
    assert data.data_quality_report(df)["duplicate_key_count"] == 0

    def wrong_symbol(symbol, start, end):
        return _bars(symbol, start, end).with_columns(pl.lit("OTHER").alias("symbol"))

    with pytest.raises(ValueError, match="other symbols"):
        data.fetch("X", "2024-01-01", "2024-01-31", source=data.CallableSource(wrong_symbol, name="wrong"), cache_dir=tmp_path)


def test_quality_failures_flags_bad_reports(tmp_path):
    ok = data.data_quality_report(data.fetch("SYN_A", "2022-01-01", "2022-01-31", cache_dir=tmp_path))
    assert data.quality_failures(ok) == []
    assert "no rows" in data.quality_failures({**ok, "row_count": 0})
    assert data.quality_failures({**ok, "duplicate_key_count": 2})
    assert data.quality_failures({**ok, "null_counts": {**ok["null_counts"], "close": 1}})


# --- adapter protocol ----------------------------------------------------------


def test_adapter_protocol():
    assert isinstance(CountingSource(), data.DataSource)
    assert isinstance(data.LocalSampleSource(), data.DataSource)
    assert isinstance(data.CallableSource(lambda s, a, b: None, name="x"), data.DataSource)
    assert not isinstance(object(), data.DataSource)
    with pytest.raises(TypeError):
        data.CallableSource("not callable", name="x")  # type: ignore[arg-type]


def test_callable_source_passes_dates_and_accepts_plain_mappings(tmp_path):
    seen = []

    def fn(symbol, start, end):
        seen.append((symbol, start, end))
        return {"date": ["2024-01-02", "2024-01-03"], "open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20]}

    src = data.CallableSource(fn, name="vendor-x", terms="https://example.invalid/terms")
    df, meta = data.fetch("ZZZ", date(2024, 1, 1), "2024-01-05", source=src, cache_dir=tmp_path, return_meta=True)
    assert seen == [("ZZZ", date(2024, 1, 1), date(2024, 1, 5))]
    assert df.height == 2
    assert df["adj_close"].to_list() == df["close"].to_list()
    assert meta["terms"] == "https://example.invalid/terms"
    assert meta["source"] == "vendor-x"


def test_local_file_source_versions_by_content(tmp_path):
    path = tmp_path / "mine.csv"
    _bars("M", date(2024, 1, 1), date(2024, 1, 31)).with_columns(pl.lit("M").alias("symbol")).write_csv(path)
    src = data.LocalFileSource(path, name="mine")
    assert src.symbols == ["M"]
    v1 = src.version
    path.write_text(path.read_text().replace("\n2024-01-02,10.0", "\n2024-01-02,10.5"))
    assert data.LocalFileSource(path, name="mine").version != v1
    with pytest.raises(FileNotFoundError):
        data.LocalFileSource(tmp_path / "missing.csv")


def test_sample_source_is_labelled_synthetic():
    src = data.LocalSampleSource()
    assert src.symbols == SAMPLE_SYMBOLS
    assert "SYNTHETIC" in src.terms
    assert src.path == data.SAMPLE_DATA_PATH
    assert data.SAMPLE_DATA_PATH.stat().st_size < 1_000_000


def test_committed_sample_matches_its_generator(tmp_path):
    out = tmp_path / "sample.csv"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "data" / "make_synthetic_sample.py"), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    assert out.read_bytes() == data.SAMPLE_DATA_PATH.read_bytes()


# --- content hash --------------------------------------------------------------


def test_hash_is_pinned_for_the_full_sample(tmp_path):
    df = data.fetch(SAMPLE_SYMBOLS, "2022-01-03", "2023-12-29", cache_dir=tmp_path)
    assert df.height == 2600
    assert data.dataset_hash(df) == FULL_SAMPLE_HASH


def test_hash_is_stable_across_row_order_column_order_and_parquet_round_trip(tmp_path):
    df = data.fetch(["SYN_A", "SYN_D"], "2022-01-01", "2022-12-31", cache_dir=tmp_path)
    h = data.dataset_hash(df)
    assert h.startswith("sha256:") and len(h) == len("sha256:") + 64
    assert data.dataset_hash(df.reverse()) == h
    assert data.dataset_hash(df.sample(fraction=1.0, shuffle=True, seed=3)) == h
    assert data.dataset_hash(df.select(list(reversed(df.columns)))) == h
    df.write_parquet(tmp_path / "x.parquet")
    assert data.dataset_hash(pl.read_parquet(tmp_path / "x.parquet")) == h


def test_changing_the_data_changes_the_hash(tmp_path):
    df = data.fetch(["SYN_A", "SYN_D"], "2022-01-01", "2022-12-31", cache_dir=tmp_path)
    h = data.dataset_hash(df)
    changed = [
        df.with_columns(pl.when(pl.int_range(pl.len()) == 17).then(pl.col("close") + 1e-9).otherwise(pl.col("close")).alias("close")),
        df.head(df.height - 1),
        pl.concat([df, df.tail(1)]),
        df.rename({"volume": "vol"}),
        df.with_columns(pl.col("volume").cast(pl.Int64)),
        df.with_columns(pl.when(pl.col("symbol") == "SYN_A").then(pl.lit("SYN_Z")).otherwise(pl.col("symbol")).alias("symbol")),
        df.with_columns(pl.col("ts") + pl.duration(days=1)),
        df.with_columns(pl.col("ts").dt.replace_time_zone("UTC")),
        df.drop("adj_close"),
    ]
    hashes = [data.dataset_hash(c) for c in changed]
    assert h not in hashes
    assert len(set(hashes)) == len(hashes)


def test_hash_handles_nulls_and_rejects_unsupported_types():
    a = pl.DataFrame({"x": [1.0, None], "s": ["a", None], "b": [True, None], "d": [date(2024, 1, 1), None]})
    b = pl.DataFrame({"x": [1.0, 0.0], "s": ["a", ""], "b": [True, False], "d": [date(2024, 1, 1), date(1970, 1, 1)]})
    assert data.dataset_hash(a) == data.dataset_hash(a.reverse())
    assert data.dataset_hash(a) != data.dataset_hash(b)
    assert data.dataset_hash(pl.DataFrame()) == data.dataset_hash(pl.DataFrame())
    with pytest.raises(TypeError, match="does not support"):
        data.dataset_hash(pl.DataFrame({"l": [[1, 2]]}))


def test_record_dataset_hash_into_manifest(tmp_path):
    df, meta = data.fetch("SYN_B", "2022-01-01", "2022-03-31", cache_dir=tmp_path, return_meta=True)
    manifest = {"run_name": "r", "dataset": {"note": "kept"}}
    out = data.record_dataset_hash(manifest, df, **meta)
    assert out is manifest
    assert manifest["dataset_hash"] == meta["dataset_hash"] == manifest["dataset"]["hash"]
    assert manifest["dataset"]["hash_version"] == data.DATASET_HASH_VERSION
    assert manifest["dataset"]["note"] == "kept"
    assert manifest["dataset"]["source"] == "openquant-synthetic-sample"
    assert manifest["dataset"]["rows"] == df.height
    json.dumps(manifest)  # serializable as-is

    assert data.record_dataset_hash({}, digest="sha256:abc")["dataset_hash"] == "sha256:abc"
    with pytest.raises(ValueError, match="does not match"):
        data.record_dataset_hash({}, df, digest="sha256:abc")
    with pytest.raises(ValueError):
        data.record_dataset_hash({})


# --- run manifests -------------------------------------------------------------


def _load_runner():
    spec = importlib.util.spec_from_file_location("oq_runner_for_hash", REPO_ROOT / "experiments" / "run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_manifests_record_the_dataset_hash(tmp_path):
    runner = _load_runner()
    cfg = REPO_ROOT / "experiments" / "configs" / "futures_oil_baseline.toml"
    grid = REPO_ROOT / "experiments" / "configs" / "futures_oil_grid.toml"

    run_dir = runner.run(cfg, tmp_path / "single")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    dataset, _ = runner._dataset_from_cfg(runner._read_config(cfg))
    expected = data.dataset_hash(runner._dataset_frame(dataset))
    assert manifest["dataset_hash"] == manifest["dataset"]["hash"] == expected

    grid_dir = runner.run_grid(cfg, grid, tmp_path / "grid")
    manifests = [json.loads((grid_dir / "run_manifest.json").read_text())]
    manifests += [json.loads((p / "run_manifest.json").read_text()) for p in grid_dir.iterdir() if p.is_dir()]
    assert len(manifests) >= 4
    assert {m["dataset_hash"] for m in manifests} == {expected}

    # Different data, different hash in the manifest.
    other = openquant.research.make_synthetic_futures_dataset(n_bars=192, seed=8)
    assert data.dataset_hash(runner._dataset_frame(other)) != expected
