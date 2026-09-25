"""Shared loaders for the `test_core_*.py` files.

They read the same files under `tests/fixtures/` that the Rust integration
tests in `crates/openquant/tests/` read, so both suites assert against one
set of reference data.
"""

import csv
import math
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def load_csv_columns(relative_path, columns, convert=float):
    """Return one list per requested column of `tests/fixtures/<relative_path>`."""
    out = {name: [] for name in columns}
    with (FIXTURES / relative_path).open("r", newline="") as f:
        for row in csv.DictReader(f):
            for name in columns:
                out[name].append(convert(row[name]))
    return [out[name] for name in columns]


def load_timestamps(relative_path, column="date_time"):
    """Timestamps as the fixture stores them, milliseconds included."""
    (raw,) = load_csv_columns(relative_path, [column], convert=str)
    return raw


def finite(values):
    return [v for v in values if math.isfinite(v)]


def finite_max(values):
    return max(finite(values))


def finite_mean(values):
    kept = finite(values)
    return sum(kept) / len(kept)


def nanmean(values):
    kept = [v for v in values if not math.isnan(v)]
    return sum(kept) / len(kept)
