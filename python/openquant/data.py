from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, overload, runtime_checkable
from urllib.parse import quote

import polars as pl

CANONICAL_OHLCV_COLUMNS = [
    "ts",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
]

_COLUMN_ALIASES = {
    "ts": "ts",
    "timestamp": "ts",
    "datetime": "ts",
    "date": "ts",
    "symbol": "symbol",
    "ticker": "symbol",
    "asset": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adj_close": "adj_close",
    "adjusted_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted close": "adj_close",
    "adj close": "adj_close",
}

_ZERO_NULL_COUNTS = {
    "ts": 0,
    "symbol": 0,
    "open": 0,
    "high": 0,
    "low": 0,
    "close": 0,
    "volume": 0,
    "adj_close": 0,
}


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _canonicalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()
    for col in df.columns:
        key = _normalize_column_name(col)
        if key in _COLUMN_ALIASES:
            target = _COLUMN_ALIASES[key]
            if target in used_targets and col != target:
                continue
            rename_map[col] = target
            used_targets.add(target)
    if rename_map:
        return df.rename(rename_map)
    return df


def _validate_required_columns(df: pl.DataFrame) -> None:
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required OHLCV columns: {', '.join(missing)}")


# Timestamp strings from the Rust bindings: "%Y-%m-%d %H:%M:%S" with an optional fractional
# second (6 digits, or 9 below a microsecond). polars infers a format from the first value only, so a column
# mixing whole and fractional seconds must be parsed with this explicit format.
TS_FORMAT = "%Y-%m-%d %H:%M:%S%.f"


def _parse_ts(expr: pl.Expr) -> pl.Expr:
    """Parse timestamp strings, keeping fractional seconds (to microseconds).

    Strings in `TS_FORMAT` are parsed with it; anything else (dates, ISO `T` forms) falls back
    to polars' format inference, as before.
    """
    return pl.coalesce(
        expr.str.strptime(pl.Datetime, TS_FORMAT, strict=False),
        expr.str.strptime(pl.Datetime, strict=False),
    )


def _prepare_ohlcv_lf(df: pl.DataFrame) -> pl.LazyFrame:
    frame = _canonicalize_columns(df)
    _validate_required_columns(frame)

    lf = frame.lazy().with_columns(
        _parse_ts(pl.col("ts").cast(pl.Utf8)),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    if "adj_close" in frame.columns:
        lf = lf.with_columns(pl.col("adj_close").cast(pl.Float64))
    else:
        lf = lf.with_columns(pl.col("close").alias("adj_close"))
    return lf.select(CANONICAL_OHLCV_COLUMNS).drop_nulls(CANONICAL_OHLCV_COLUMNS)


def _format_ts(v: Any) -> str | None:
    if v is None:
        return None
    if hasattr(v, "strftime"):
        # Same form as str(datetime) and the Rust bindings: no fraction on a whole second.
        out = v.strftime("%Y-%m-%d %H:%M:%S")
        if getattr(v, "microsecond", 0):
            out += f".{v.microsecond:06d}"
        return out
    return str(v)


_DAY_US = 86_400 * 1_000_000
_HOUR_US = 3_600 * 1_000_000


def _inferred_interval_us(sorted_df: pl.DataFrame) -> int | None:
    """The bar spacing of a frame sorted by (symbol, ts_us): the most common positive spacing
    between consecutive bars of one symbol, pooled over symbols (the smallest on a tie).

    `None` when no symbol has two bars at different times. Mirrors
    `openquant::data_processing` in Rust.
    """
    spacings = (
        sorted_df.select(
            pl.when(pl.col("symbol") == pl.col("symbol").shift(1))
            .then(pl.col("ts_us") - pl.col("ts_us").shift(1))
            .alias("spacing")
        )
        .get_column("spacing")
        .drop_nulls()
    )
    spacings = spacings.filter(spacings > 0)
    if spacings.len() == 0:
        return None
    counts = spacings.value_counts(name="n")
    top = counts.filter(pl.col("n") == pl.col("n").max())
    return int(top.get_column("spacing").min())  # type: ignore[arg-type]


def _gap_count(sorted_df: pl.DataFrame, interval_us: int | None) -> int:
    """Gaps between consecutive bars of one symbol, given the inferred bar spacing.

    Daily data (spacing within an hour of one day, to absorb DST shifts): a gap is a skipped
    weekday, i.e. at least one Monday-to-Friday date strictly between the two bars' dates,
    so weekends are not gaps (exchange holidays still are). Any other spacing: a gap is a
    spacing greater than the inferred one.
    """
    if interval_us is None:
        return 0
    same = pl.col("symbol") == pl.col("symbol").shift(1)
    if abs(interval_us - _DAY_US) <= _HOUR_US:
        prev_date = pl.col("ts").shift(1).dt.date()
        cur_date = pl.col("ts").dt.date()
        gap = same & (
            pl.business_day_count(prev_date + pl.duration(days=1), cur_date).fill_null(0) > 0
        )
    else:
        gap = same & ((pl.col("ts_us") - pl.col("ts_us").shift(1)) > interval_us)
    return int(sorted_df.select(gap.fill_null(False).cast(pl.UInt32).sum()).item())


def _build_quality_report(
    sorted_df: pl.DataFrame, rows_removed_by_deduplication: int
) -> dict[str, Any]:
    if sorted_df.height == 0:
        return {
            "row_count": 0,
            "symbol_count": 0,
            "duplicate_key_count": 0,
            "gap_interval_count": 0,
            "inferred_interval_us": None,
            "ts_min": None,
            "ts_max": None,
            "rows_removed_by_deduplication": rows_removed_by_deduplication,
            "null_counts": dict(_ZERO_NULL_COUNTS),
        }

    interval_us = _inferred_interval_us(sorted_df)

    summary = (
        sorted_df.lazy()
        .select(
            pl.len().alias("row_count"),
            pl.col("symbol").n_unique().alias("symbol_count"),
            (
                (
                    (pl.col("symbol") == pl.col("symbol").shift(1))
                    & (pl.col("ts_us") == pl.col("ts_us").shift(1))
                )
                .cast(pl.UInt32)
                .sum()
            ).alias("duplicate_key_count"),
            pl.col("ts").min().alias("ts_min"),
            pl.col("ts").max().alias("ts_max"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "row_count": int(summary["row_count"]),
        "symbol_count": int(summary["symbol_count"]),
        "duplicate_key_count": int(summary["duplicate_key_count"]),
        "gap_interval_count": _gap_count(sorted_df, interval_us),
        "inferred_interval_us": interval_us,
        "ts_min": _format_ts(summary["ts_min"]),
        "ts_max": _format_ts(summary["ts_max"]),
        "rows_removed_by_deduplication": rows_removed_by_deduplication,
        "null_counts": dict(_ZERO_NULL_COUNTS),
    }


def _interval_to_seconds(interval: str) -> int:
    s = interval.strip().lower()
    if s.endswith("d"):
        return int(s[:-1]) * 24 * 3600
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("s"):
        return int(s[:-1])
    raise ValueError(f"unsupported interval format: {interval}")


@overload
def clean_ohlcv(
    df: pl.DataFrame,
    *,
    dedupe_keep: Literal["first", "last"] = ...,
    return_report: Literal[False] = ...,
) -> pl.DataFrame: ...


@overload
def clean_ohlcv(
    df: pl.DataFrame,
    *,
    dedupe_keep: Literal["first", "last"] = ...,
    return_report: Literal[True],
) -> tuple[pl.DataFrame, dict[str, Any]]: ...


@overload
def clean_ohlcv(
    df: pl.DataFrame,
    *,
    dedupe_keep: Literal["first", "last"] = ...,
    return_report: bool = ...,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]: ...


def clean_ohlcv(
    df: pl.DataFrame,
    *,
    dedupe_keep: Literal["first", "last"] = "last",
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    if dedupe_keep not in {"first", "last"}:
        raise ValueError("dedupe_keep must be 'first' or 'last'")

    base_lf = _prepare_ohlcv_lf(df).with_columns(
        pl.col("ts").dt.timestamp(time_unit="us").alias("ts_us")
    )
    sorted_lf = base_lf.sort(["symbol", "ts_us"])

    duplicate_key_count = int(
        sorted_lf.select(
            (
                (
                    (pl.col("symbol") == pl.col("symbol").shift(1))
                    & (pl.col("ts_us") == pl.col("ts_us").shift(1))
                )
                .cast(pl.UInt32)
                .sum()
            ).alias("duplicate_key_count")
        )
        .collect()
        .item(0, 0)
    )

    cleaned = (
        sorted_lf.unique(
            subset=["symbol", "ts_us"],
            keep=dedupe_keep,
            maintain_order=True,
        )
        .sort(["symbol", "ts"])
        .collect()
    )

    frame = cleaned.select(CANONICAL_OHLCV_COLUMNS)
    if not return_report:
        return frame

    report = _build_quality_report(cleaned, rows_removed_by_deduplication=duplicate_key_count)
    report["duplicate_key_count"] = 0
    return frame, report


def data_quality_report(df: pl.DataFrame) -> dict[str, Any]:
    sorted_df = (
        _prepare_ohlcv_lf(df)
        .with_columns(pl.col("ts").dt.timestamp(time_unit="us").alias("ts_us"))
        .sort(["symbol", "ts_us"])
        .collect()
    )
    return _build_quality_report(sorted_df, rows_removed_by_deduplication=0)


def load_ohlcv(
    path: str | Path,
    *,
    symbol: str | None = None,
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        raw = pl.read_csv(file_path)
    elif suffix in {".parquet", ".pq"}:
        raw = pl.read_parquet(file_path)
    else:
        raise ValueError(f"unsupported file type: {suffix}")

    raw = _canonicalize_columns(raw)
    if "symbol" not in raw.columns:
        if symbol is None:
            raise ValueError("symbol column missing and no symbol argument provided")
        raw = raw.with_columns(pl.lit(symbol).alias("symbol"))
    return clean_ohlcv(raw, return_report=return_report)


@overload
def align_calendar(
    df: pl.DataFrame,
    *,
    interval: str = ...,
    return_report: Literal[False] = ...,
) -> pl.DataFrame: ...


@overload
def align_calendar(
    df: pl.DataFrame,
    *,
    interval: str = ...,
    return_report: Literal[True],
) -> tuple[pl.DataFrame, dict[str, Any]]: ...


@overload
def align_calendar(
    df: pl.DataFrame,
    *,
    interval: str = ...,
    return_report: bool = ...,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]: ...


def align_calendar(
    df: pl.DataFrame,
    *,
    interval: str = "1d",
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    """Clean ``df`` and reindex each symbol onto a regular grid of ``interval``.

    Each symbol's grid runs from its first to its last bar in steps of ``interval``. Grid
    points with no bar get null prices and ``is_missing_bar=True``. A bar whose timestamp is
    not on its symbol's grid (for example a daily bar stamped at a different time of day)
    is **not** in the output. With ``return_report=True`` the result is ``(aligned, report)``
    and ``report["off_grid_bars"]`` holds those bars; otherwise a ``UserWarning`` names how
    many were dropped. ``report`` also carries ``off_grid_bar_count`` and
    ``rows_removed_by_deduplication``.
    """
    interval_seconds = _interval_to_seconds(interval)
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    clean_df, clean_report = clean_ohlcv(df, return_report=True)
    clean = clean_df.lazy()
    bounds = clean.group_by("symbol").agg(
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
    )
    calendar = (
        bounds.with_columns(
            pl.datetime_ranges(
                "ts_min",
                "ts_max",
                interval=f"{interval_seconds}s",
                closed="both",
            ).alias("ts")
        )
        .explode("ts")
        .select(["symbol", "ts"])
    )

    out = (
        calendar.join(clean, on=["symbol", "ts"], how="left")
        .with_columns(pl.col("open").is_null().alias("is_missing_bar"))
        .select(CANONICAL_OHLCV_COLUMNS + ["is_missing_bar"])
        .sort(["symbol", "ts"])
        .collect()
    )
    off_grid = (
        clean.join(calendar, on=["symbol", "ts"], how="anti")
        .select(CANONICAL_OHLCV_COLUMNS)
        .sort(["symbol", "ts"])
        .collect()
    )
    if return_report:
        report = {
            "rows_removed_by_deduplication": clean_report["rows_removed_by_deduplication"],
            "off_grid_bar_count": off_grid.height,
            "off_grid_bars": off_grid,
        }
        return out, report
    if off_grid.height:
        warnings.warn(
            f"align_calendar dropped {off_grid.height} bar(s) whose timestamp is not on "
            f"their symbol's {interval} grid; pass return_report=True to get them",
            UserWarning,
            stacklevel=2,
        )
    return out


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

DATASET_HASH_VERSION = "oq-dataset-sha256-v1"


def _hash_kind(name: str, dtype: pl.DataType) -> tuple[str, pl.Expr]:
    """Map a column to (kind label, expression yielding hashable Python values)."""
    col = pl.col(name)
    if dtype == pl.Boolean:
        return "bool", col
    if dtype.is_integer():
        return "int", col
    if dtype.is_float():
        return "float", col.cast(pl.Float64)
    if dtype == pl.Utf8 or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return "str", col.cast(pl.Utf8)
    if isinstance(dtype, pl.Datetime):
        tz = dtype.time_zone or "naive"
        return f"datetime[us,{tz}]", col.dt.epoch("us")
    if dtype == pl.Date:
        return "date", col.cast(pl.Int32)
    raise TypeError(f"dataset_hash does not support column {name!r} of type {dtype}")


def _hash_field(value: Any) -> str:
    if value is None:
        return "\\N"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def dataset_hash(df: pl.DataFrame) -> str:
    """Deterministic content hash of a table, as ``"sha256:<64 hex>"``.

    The hash describes the data, not its layout. Definition
    (``oq-dataset-sha256-v1``):

    1. Columns are taken in sorted name order.
    2. Each column is mapped to a kind: ``int``, ``float``, ``str``, ``bool``,
       ``date`` (days since the Unix epoch) or ``datetime[us,<tz>]``
       (microseconds since the Unix epoch). Other types raise ``TypeError``.
    3. Rows are sorted by every column in that order (nulls last), so input row
       order does not matter; duplicate rows count.
    4. The SHA-256 input is the line ``oq-dataset-sha256-v1``, one line
       ``"<name>":<kind>`` per column, then one line per row of tab-separated
       fields: floats as Python ``repr`` (exact round-trip), strings as JSON,
       integers in decimal, booleans as ``true``/``false`` and nulls as ``\\N``.

    It does not depend on Parquet/Arrow byte layout or the Polars version. Any
    change to a value, a column name, a column type or the set of rows changes it.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("dataset_hash expects a polars DataFrame")
    cols = sorted(df.columns)
    kinds: list[str] = []
    exprs: list[pl.Expr] = []
    for name in cols:
        kind, expr = _hash_kind(name, df.schema[name])
        kinds.append(kind)
        exprs.append(expr.alias(name))
    canonical = df.select(exprs)
    if cols and canonical.height:
        canonical = canonical.sort(cols, nulls_last=True, maintain_order=True)

    h = hashlib.sha256()
    h.update((DATASET_HASH_VERSION + "\n").encode("utf-8"))
    for name, kind in zip(cols, kinds):
        h.update(f"{json.dumps(name, ensure_ascii=False)}:{kind}\n".encode())
    for row in canonical.iter_rows():
        h.update(("\t".join(_hash_field(v) for v in row) + "\n").encode("utf-8"))
    return "sha256:" + h.hexdigest()


def record_dataset_hash(
    manifest: dict[str, Any],
    frame: pl.DataFrame | None = None,
    *,
    digest: str | None = None,
    **provenance: Any,
) -> dict[str, Any]:
    """Record a dataset's content hash, and its provenance, in a run manifest.

    Sets ``manifest["dataset_hash"]`` and merges ``hash``, ``hash_version``,
    ``rows`` and any ``provenance`` keywords (for example the metadata returned
    by ``fetch(..., return_meta=True)``) into ``manifest["dataset"]``. Pass the
    ``frame``, or a ``digest`` already computed by :func:`dataset_hash`.
    Returns ``manifest``, modified in place.
    """
    if frame is None and digest is None:
        raise ValueError("pass a frame or a digest")
    value = dataset_hash(frame) if frame is not None else digest
    if digest is not None and value != digest:
        raise ValueError("digest does not match frame")
    dataset = dict(manifest.get("dataset") or {})
    dataset.update({k: v for k, v in provenance.items() if k != "dataset_hash"})
    dataset["hash"] = value
    dataset["hash_version"] = DATASET_HASH_VERSION
    if frame is not None:
        dataset["rows"] = frame.height
    manifest["dataset"] = dataset
    manifest["dataset_hash"] = value
    return manifest


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "_sample_data" / "synthetic_daily_ohlcv.csv"
SAMPLE_TERMS = (
    "SYNTHETIC data generated by scripts/data/make_synthetic_sample.py; not market data; "
    "MIT (repository license). See DATA_SOURCES.md."
)


@runtime_checkable
class DataSource(Protocol):
    """What ``fetch`` needs from a market data source.

    ``name`` identifies the source in the cache path and the run manifest; keep
    it stable. ``fetch_symbol`` returns daily bars for one symbol with dates in
    ``[start, end]`` (inclusive), as a Polars DataFrame or anything
    ``polars.DataFrame(...)`` accepts, with any column names ``clean_ohlcv``
    recognizes (``date``/``ts``, ``open``, ``high``, ``low``, ``close``,
    ``volume``, optional ``adj_close`` and ``symbol``).

    Optional attributes: ``version`` (a string; changing it starts a new cache
    namespace for the source) and ``terms`` (a URL or statement of the data's
    terms, copied into ``fetch``'s metadata so it lands in the run manifest).
    """

    name: str

    def fetch_symbol(self, symbol: str, start: date, end: date) -> Any: ...


class LocalFileSource:
    """A source backed by one local CSV/Parquet OHLCV file with a ``symbol`` column.

    ``version`` is derived from the file's bytes, so editing the file starts a
    new cache namespace instead of serving stale bars.
    """

    def __init__(
        self, path: str | Path, *, name: str = "local-file", terms: str | None = None
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"file not found: {self.path}")
        self.name = name
        self.terms = (
            terms
            if terms is not None
            else f"local file {self.path.name}; terms are the file owner's"
        )
        self.version = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        self._frame: pl.DataFrame | None = None

    def _data(self) -> pl.DataFrame:
        if self._frame is None:
            frame = load_ohlcv(self.path)
            assert isinstance(frame, pl.DataFrame)
            self._frame = frame
        return self._frame

    @property
    def symbols(self) -> list[str]:
        return sorted(self._data()["symbol"].unique().to_list())

    def fetch_symbol(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        if symbol not in self.symbols:
            raise LookupError(
                f"symbol {symbol!r} is not in {self.path.name}; available: {', '.join(self.symbols)}"
            )
        return self._data().filter(pl.col("symbol") == symbol)


class LocalSampleSource(LocalFileSource):
    """The SYNTHETIC daily OHLCV sample shipped with openquant.

    Symbols ``SYN_A`` ... ``SYN_E`` (not real tickers), Monday-Friday from
    2022-01-03 to 2023-12-29, generated by ``scripts/data/make_synthetic_sample.py``
    from a fixed seed. It is not market data; it exists so tests, docs and
    runbooks run offline.
    """

    def __init__(self) -> None:
        super().__init__(SAMPLE_DATA_PATH, name="openquant-synthetic-sample", terms=SAMPLE_TERMS)


class CallableSource:
    """Adapt a function ``fn(symbol, start, end) -> frame`` into a ``DataSource``.

    This is the pattern for user-supplied, fetch-only adapters: ``fn`` calls the
    user's own vendor account, reading the user's own key from an environment
    variable inside ``fn`` (never pass or store the key here). Fetched data is
    cached on the user's machine and is never part of the repository.
    """

    def __init__(
        self,
        fn: Callable[[str, date, date], Any],
        *,
        name: str,
        version: str = "",
        terms: str | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn
        self.name = name
        self.version = version
        self.terms = terms

    def fetch_symbol(self, symbol: str, start: date, end: date) -> Any:
        return self.fn(symbol, start, end)


# ---------------------------------------------------------------------------
# fetch + on-disk cache
# ---------------------------------------------------------------------------

CACHE_FORMAT = "openquant-ohlcv-cache-v1"


class CacheMissError(LookupError):
    """Raised by ``fetch(..., offline=True)`` when a request is not in the cache."""


def default_cache_dir() -> Path:
    """``$OPENQUANT_DATA_CACHE``, else ``$XDG_CACHE_HOME/openquant/data``, else ``~/.cache/openquant/data``."""
    env = os.environ.get("OPENQUANT_DATA_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "openquant" / "data"


def _as_date(value: date | datetime | str, what: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise ValueError(f"{what} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc
    raise TypeError(f"{what} must be a date, datetime or ISO date string")


def _as_symbols(symbols: str | Iterable[str]) -> list[str]:
    items = [symbols] if isinstance(symbols, str) else list(symbols)
    out: list[str] = []
    for s in items:
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"symbols must be non-empty strings, got {s!r}")
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("symbols must not be empty")
    return out


def _source_dir(source: Any) -> str:
    name = getattr(source, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("source.name must be a non-empty string")
    version = str(getattr(source, "version", "") or "")
    key = quote(name, safe="")
    return f"{key}@{quote(version, safe='')}" if version else key


def _cache_paths(root: Path, source: Any, symbol: str, start: date, end: date) -> tuple[Path, Path]:
    base = root / _source_dir(source) / quote(symbol, safe="")
    stem = f"{start.isoformat()}_{end.isoformat()}"
    return base / f"{stem}.parquet", base / f"{stem}.json"


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _normalize_source_frame(
    raw: Any, source_name: str, symbol: str, start: date, end: date
) -> pl.DataFrame:
    frame = raw if isinstance(raw, pl.DataFrame) else pl.DataFrame(raw)
    frame = _canonicalize_columns(frame)
    if "symbol" in frame.columns:
        if frame.filter(pl.col("symbol").cast(pl.Utf8) != symbol).height:
            raise ValueError(
                f"source {source_name!r} returned rows for other symbols when asked for {symbol!r}"
            )
    else:
        frame = frame.with_columns(pl.lit(symbol).alias("symbol"))
    cleaned = clean_ohlcv(frame)
    assert isinstance(cleaned, pl.DataFrame)
    cleaned = cleaned.filter(pl.col("ts").dt.date().is_between(start, end, closed="both"))
    if cleaned.height == 0:
        raise ValueError(
            f"source {source_name!r} returned no usable rows for {symbol!r} between {start} and {end}"
        )
    return cleaned


def quality_failures(report: Mapping[str, Any]) -> list[str]:
    """Reasons a :func:`data_quality_report` fails ``fetch``'s bar; empty when it passes.

    The bar: at least one row, no duplicate (symbol, ts) keys, no nulls. Gaps are
    reported but not failed, since daily data has weekends and holidays.
    """
    failures = []
    if report["row_count"] == 0:
        failures.append("no rows")
    if report["duplicate_key_count"]:
        failures.append(f"{report['duplicate_key_count']} duplicate (symbol, ts) keys")
    nulls = {k: v for k, v in report["null_counts"].items() if v}
    if nulls:
        failures.append(f"nulls in {nulls}")
    return failures


def fetch(
    symbols: str | Iterable[str],
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    source: DataSource | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    offline: bool = False,
    return_meta: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    """Fetch daily OHLCV bars for ``symbols`` over ``[start, end]`` through a local cache.

    Each (source, symbol, start, end) request is cached as one Parquet file (plus
    a JSON sidecar) under ``cache_dir``, default :func:`default_cache_dir`. A
    repeated request is read from disk without calling the source, so it works
    offline. ``source`` defaults to :class:`LocalSampleSource`, which is
    SYNTHETIC data. ``refresh=True`` refetches; ``offline=True`` raises
    :class:`CacheMissError` instead of calling the source.

    Returns the canonical frame of :func:`clean_ohlcv` (``ts, symbol, open,
    high, low, close, volume, adj_close``, sorted by symbol then ts), which has
    passed :func:`data_quality_report` (see :func:`quality_failures`). With
    ``return_meta=True`` it also returns provenance for a run manifest: the
    source's name, version and terms, the request, each symbol's cache status
    and the frame's :func:`dataset_hash`.
    """
    syms = _as_symbols(symbols)
    start_d = _as_date(start, "start")
    end_d = _as_date(end, "end")
    if start_d > end_d:
        raise ValueError(f"start {start_d} is after end {end_d}")
    src: Any = LocalSampleSource() if source is None else source
    if not isinstance(src, DataSource):
        raise TypeError(
            "source must have a `name` string and a `fetch_symbol(symbol, start, end)` method"
        )
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()

    frames: list[pl.DataFrame] = []
    cache_status: dict[str, str] = {}
    for sym in syms:
        data_path, meta_path = _cache_paths(root, src, sym, start_d, end_d)
        frame: pl.DataFrame | None = None
        if not refresh and data_path.exists() and meta_path.exists():
            cached = pl.read_parquet(data_path)
            recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("dataset_hash")
            if recorded == dataset_hash(cached):
                frame = cached
                cache_status[sym] = "hit"
            elif offline:
                raise CacheMissError(f"cache entry {data_path} does not match its recorded hash")
        if frame is None:
            if offline:
                raise CacheMissError(
                    f"{sym!r} {start_d}..{end_d} from {src.name!r} is not cached in {root}"
                )
            frame = _normalize_source_frame(
                src.fetch_symbol(sym, start_d, end_d), src.name, sym, start_d, end_d
            )
            entry = {
                "format": CACHE_FORMAT,
                "source": src.name,
                "source_version": str(getattr(src, "version", "") or ""),
                "terms": getattr(src, "terms", None),
                "symbol": sym,
                "start": start_d.isoformat(),
                "end": end_d.isoformat(),
                "rows": frame.height,
                "dataset_hash": dataset_hash(frame),
                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            _atomic_write(data_path, frame.write_parquet)

            def _write_meta(p: Path, e: dict[str, Any] = entry) -> None:
                p.write_text(json.dumps(e, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            _atomic_write(meta_path, _write_meta)
            cache_status[sym] = "refresh" if refresh else "miss"
        frames.append(frame)

    out = pl.concat(frames, how="vertical").sort(["symbol", "ts"]).select(CANONICAL_OHLCV_COLUMNS)
    failures = quality_failures(data_quality_report(out))
    if failures:
        raise ValueError(f"fetched data failed data_quality_report: {'; '.join(failures)}")
    if not return_meta:
        return out
    meta = {
        "source": src.name,
        "source_version": str(getattr(src, "version", "") or ""),
        "terms": getattr(src, "terms", None),
        "symbols": syms,
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "rows": out.height,
        "cache_dir": str(root),
        "cache": cache_status,
        "dataset_hash": dataset_hash(out),
    }
    return out, meta
