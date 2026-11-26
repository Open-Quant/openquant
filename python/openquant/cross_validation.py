"""Purged k-fold, walk-forward and combinatorial purged cross-validation (AFML chapters 7, 12).

Every splitter returns indices, never fits a model: pass the splits to any estimator, for
example ``sklearn.model_selection.cross_val_score(model, X, y, cv=splits)``. The splitting,
purging and embargo are the Rust ``openquant::cross_validation`` code.

**Features with memory.** Purged k-fold and CPCV fit the model for a middle fold on samples
*after* it as well. Purging and the embargo remove overlapping labels, not that: a feature that
carries the price level lets the model learn where the path went after the test fold, and it
scores above chance on random walks (issue #217). Compare such features with
:func:`walk_forward_splits`, and against :func:`null_score_distribution`, a no-signal null run
through the same splits.

Label spans are required. ``t0[i]`` is when sample ``i``'s label starts (usually the event
time) and ``t1[i]`` when it is resolved (the triple-barrier touch). They may be numpy
``datetime64`` arrays, pandas or polars datetime columns, lists of ``datetime`` or ISO strings,
or plain integers such as bar positions; samples must be in time order. Integers must
fit in signed int64. Datetimes must fit in signed int64 nanoseconds (excluding NaT);
out-of-range dates are rejected rather than silently wrapped. Masked label spans and
indices are rejected, and count arguments must be integers, not booleans or floats.
Sub-nanosecond values
are floored to nanoseconds, as in NumPy, so distinct times can become equal.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import numpy as np

from . import _core

_cv = _core.cross_validation

__all__ = [
    "purged_kfold_splits",
    "split_with_diagnostics",
    "cpcv_splits",
    "cpcv_paths",
    "walk_forward_splits",
    "walk_forward_split_with_diagnostics",
    "null_score_distribution",
    "null_p_value",
    "bootstrap_returns",
    "naive_kfold_splits",
    "count_train_test_overlaps",
]

_INDEX_KEYS = ("train_indices", "test_indices", "purged_indices", "embargo_indices")


def _datetime_ns(arr: np.ndarray, name: str) -> np.ndarray:
    """Convert datetime counts without NumPy's overflowing unit arithmetic."""
    if np.isnat(arr).any():
        raise ValueError(f"{name} contains NaT")
    ticks = arr.astype(np.int64, copy=False)  # unit counts, with native byte order
    unit, multiplier = np.datetime_data(arr.dtype)
    lo, hi = -(2**63) + 1, 2**63 - 1
    error = f"{name} is outside the supported int64 nanosecond range"
    if unit in ("Y", "M"):
        # Calendar units have variable length. Bound the year before NumPy's
        # calendar conversion, then use exact fixed-unit arithmetic below.
        years = [1970 + int(tick) * multiplier // (12 if unit == "M" else 1) for tick in ticks]
        if any(year < 1677 or year > 2262 for year in years):
            raise ValueError(error)
        return _datetime_ns(arr.astype("datetime64[D]"), name)
    fine_scale = {"ps": 1000, "fs": 1_000_000, "as": 1_000_000_000}.get(unit)
    if fine_scale is not None:
        if multiplier == 1:
            # astype can overflow its negative-floor adjustment near int64 min.
            return ticks // fine_scale
        exact = [int(tick) * multiplier // fine_scale for tick in ticks]
        if any(value < lo or value > hi for value in exact):
            raise ValueError(error)
        return np.asarray(exact, dtype=np.int64)
    scale = {
        "W": 604800_000000000,
        "D": 86400_000000000,
        "h": 3600_000000000,
        "m": 60_000000000,
        "s": 1_000000000,
        "ms": 1_000000,
        "us": 1000,
        "ns": 1,
    }.get(unit)
    if scale is None:
        if ticks.size == 0:
            return ticks
        raise ValueError(f"{name} needs a datetime unit")
    factor = multiplier * scale
    if (ticks < -((-lo) // factor)).any() or (ticks > hi // factor).any():
        raise ValueError(error)
    # A factor beyond int64 only permits zero ticks. Avoid object promotion.
    if factor > hi:
        return np.zeros(ticks.shape, dtype=np.int64)
    return ticks * factor if factor != 1 else ticks


def _object_datetime_ns(values: Any, name: str) -> np.ndarray:
    """Parse each original date before mixed precisions can silently wrap."""
    result = []
    for value in np.asarray(values, dtype=object):
        if isinstance(value, datetime):
            # pandas Timestamp is a datetime subclass, but NumPy's generic
            # datetime conversion reads only its Python microseconds field.
            # Use the scalar protocol without requiring pandas for other callers.
            to_datetime64 = getattr(value, "to_datetime64", None)
            if callable(to_datetime64):
                converted = to_datetime64()
                if isinstance(converted, np.datetime64):
                    value = converted
        if isinstance(value, np.datetime64):
            result.append(int(_datetime_ns(np.asarray([value]), name)[0]))
            continue
        if isinstance(value, (str, bytes)):
            text = value.decode("ascii") if isinstance(value, bytes) else value
            year = re.match(r"^\s*([+-]?\d+)", text)
            if year is not None and not 1677 <= int(year[1]) <= 2262:
                raise ValueError(f"{name} is outside the supported int64 nanosecond range")
        if isinstance(value, (int, float, complex, bool)):
            raise ValueError(f"{name} object values must hold dates, not numeric timestamps")
        try:
            # Explicit units avoid inferring ps/fs/as for modern ISO strings,
            # which overflows before a later conversion can detect it.
            day = np.datetime64(value, "D")
            ns = np.datetime64(value, "ns")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must hold datetimes, ISO timestamp strings or integers: {exc}"
            ) from None
        if np.isnat(day) or np.isnat(ns):
            raise ValueError(f"{name} contains NaT or is outside the int64 nanosecond range")
        count = int(ns.astype(np.int64))
        # Use integer floor, not ns.astype('datetime64[D]'): that conversion
        # itself overflows near the negative int64 boundary in NumPy.
        if count // 86400_000000000 != int(day.astype(np.int64)):
            raise ValueError(f"{name} is outside the supported int64 nanosecond range")
        result.append(count)
    return np.asarray(result, dtype=np.int64)


def _mixed_int64(values: Any, arr: np.ndarray, name: str) -> np.ndarray | None:
    """Recover signed/unsigned integer mixtures before NumPy's lossy promotion."""
    if not arr.size or arr.dtype.kind not in "fO":
        return None
    if isinstance(values, np.ndarray) and arr.dtype.kind != "O":
        return None
    original = np.asarray(values, dtype=object)
    if not all(
        (isinstance(value, int) and not isinstance(value, bool))
        or (isinstance(value, np.integer) and value.dtype.kind in "iu")
        for value in original
    ):
        return None
    exact = [int(value) for value in original]
    if any(not -(2**63) <= value <= (2**63 - 1) for value in exact):
        raise ValueError(f"{name} must fit in int64 without overflow")
    return np.asarray(exact, dtype=np.int64)


def _unmasked_asarray(value: Any, name: str) -> np.ndarray:
    """Reject missing observations before NumPy drops a MaskedArray's mask."""
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmask(value)):
        raise ValueError(f"{name} contains masked values")
    return np.asarray(value)


def _count(value: Any, name: str) -> int:
    """Accept lossless integer indices, not int() coercions."""
    if np.ma.is_masked(value):
        raise ValueError(f"{name} contains a masked value")
    if not isinstance(value, (bool, np.bool_)):
        try:
            return operator.index(value)
        except TypeError:
            pass
    raise ValueError(f"{name} must be an integer")


def _as_int64(values: Any, name: str) -> tuple[np.ndarray, bool]:
    """One span column as int64, and whether it held datetimes.

    Datetimes become nanoseconds since the epoch; integers pass through unchanged.
    """
    if values is None:
        raise ValueError(
            f"{name} is required: purging needs every label's span (t0, t1). Without it, "
            "training labels that overlap the test fold cannot be found."
        )
    arr = _unmasked_asarray(values, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    exact = _mixed_int64(values, arr, name)
    if exact is not None:
        return exact, False
    if arr.dtype.kind in "OUS" or (arr.dtype.kind == "M" and not hasattr(values, "dtype")):
        return _object_datetime_ns(values, name), True
    if arr.dtype.kind == "M":
        return _datetime_ns(arr, name), True
    if arr.dtype.kind in "iu" or arr.size == 0:
        if not hasattr(values, "dtype") and any(
            isinstance(value, (bool, np.bool_)) for value in values
        ):
            raise ValueError(f"{name} must hold integer positions, not boolean values")
        if arr.dtype.kind == "u" and (arr > np.iinfo(np.int64).max).any():
            raise ValueError(f"{name} must fit in int64 without overflow")
        return arr.astype(np.int64), False
    raise ValueError(f"{name} must hold datetimes or integers, got dtype {arr.dtype}")


def _label_spans(t0: Any, t1: Any) -> tuple[list[int], list[int]]:
    """Both span columns as int64 lists of equal length and the same kind."""
    s0, t0_is_time = _as_int64(t0, "t0")
    s1, t1_is_time = _as_int64(t1, "t1")
    if len(s0) and len(s1) and t0_is_time != t1_is_time:
        raise ValueError("t0 and t1 must both be datetimes or both be integers")
    if len(s0) != len(s1):
        raise ValueError(f"t0/t1 length mismatch: {len(s0)} vs {len(s1)}")
    return s0.tolist(), s1.tolist()


def _index_list(values: Sequence[int], name: str) -> list[int]:
    arr = _unmasked_asarray(values, name)
    error = f"{name} must be a one-dimensional array of integers"
    if arr.ndim != 1:
        raise ValueError(error)
    if arr.size == 0:
        return []
    if arr.dtype.kind not in "iu":
        exact = _mixed_int64(values, arr, name)
        if exact is None:
            raise ValueError(error)
        arr = exact
    if not isinstance(values, np.ndarray) and any(
        isinstance(value, (bool, np.bool_)) for value in values
    ):
        raise ValueError(error)
    if (arr < 0).any():
        raise ValueError(f"{name} must be non-negative")
    if arr.dtype.kind == "u" and (arr > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name} must fit in int64 without overflow")
    return arr.astype(np.int64).tolist()


def _indices(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.intp)


def _split_dict(d: dict) -> dict:
    out = dict(d)
    for key in _INDEX_KEYS:
        out[key] = _indices(out[key])
    out["test_ranges"] = [tuple(r) for r in out["test_ranges"]]
    if "test_fold_ids" in out:
        out["test_fold_ids"] = tuple(out["test_fold_ids"])
    return out


def purged_kfold_splits(
    t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged k-fold splits (AFML Snippet 7.3) as ``(train_idx, test_idx)`` numpy arrays.

    Folds are contiguous blocks in sample order. A training label is purged when its span
    intersects the test fold's window, from its first label's start to its latest label end.
    ``ceil(pct_embargo * n)`` further samples are embargoed after the fold only, starting where
    the purge ends: at the first sample whose label starts after the fold's latest label end.
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        (_indices(train), _indices(test))
        for train, test in _cv.purged_kfold_splits(
            s0, s1, _count(n_splits, "n_splits"), float(pct_embargo)
        )
    ]


def split_with_diagnostics(t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0) -> list[dict]:
    """The folds of :func:`purged_kfold_splits`, each with why every excluded sample left.

    Each dict has ``split_id``, ``train_indices``, ``test_indices``, ``test_ranges``
    (half-open ``(start, stop)`` pairs), ``purged_indices``, ``embargo_indices`` (every
    non-test sample in an embargo window, purged or not) and ``overlap_count_after_purge``
    (always 0).
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        _split_dict(d)
        for d in _cv.split_with_diagnostics(
            s0, s1, _count(n_splits, "n_splits"), float(pct_embargo)
        )
    ]


def cpcv_splits(
    t0: Any, t1: Any, n_splits: int, n_test_splits: int, pct_embargo: float = 0.0
) -> list[dict]:
    """Combinatorial purged CV splits (AFML §12.4): one per choice of ``n_test_splits`` folds.

    There are C(n_splits, n_test_splits) splits, in lexicographic order of ``test_fold_ids``.
    Each dict has the keys of :func:`split_with_diagnostics` plus ``test_fold_ids``. Adjacent
    test folds are purged as one block, so ``n_test_splits = 1`` gives the k-fold splits.
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        _split_dict(d)
        for d in _cv.cpcv_splits(
            s0,
            s1,
            _count(n_splits, "n_splits"),
            _count(n_test_splits, "n_test_splits"),
            float(pct_embargo),
        )
    ]


def cpcv_paths(n_splits: int, n_test_splits: int) -> np.ndarray:
    """The φ = k/N · C(N, k) CPCV backtest paths as an ``(n_paths, n_splits)`` int array.

    ``paths[p, g]`` is the ``split_id`` of the split whose predictions path ``p`` uses for fold
    ``g``: the ``p``-th split, in ``split_id`` order, that tests fold ``g`` (AFML §12.4). The
    numbering matches :func:`openquant.backtesting_engine.run_cpcv`.
    """
    return np.asarray(
        _cv.cpcv_paths(_count(n_splits, "n_splits"), _count(n_test_splits, "n_test_splits")),
        dtype=np.intp,
    )


def walk_forward_splits(
    t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0, min_train_folds: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward splits over the folds of :func:`purged_kfold_splits`, forward-only.

    Fold ``g >= min_train_folds`` is tested exactly as in :func:`purged_kfold_splits`, but it
    trains only on the samples *before* it whose label spans do not reach its window (purged).
    No model sees a sample after its test fold, so the embargo, which only drops samples after
    a test block, removes nothing. The first ``min_train_folds`` folds are never tested.

    Use it to compare features with memory (price levels, weakly differenced prices, long
    averages): purged k-fold and CPCV inflate them even on random walks (issue #217). Because
    the folds are the k-fold folds, a feature can be scored fold for fold under both schemes.
    """
    return [
        (d["train_indices"], d["test_indices"])
        for d in walk_forward_split_with_diagnostics(t0, t1, n_splits, pct_embargo, min_train_folds)
    ]


def walk_forward_split_with_diagnostics(
    t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0, min_train_folds: int = 1
) -> list[dict]:
    """The splits of :func:`walk_forward_splits` as dicts, with the purged indices.

    Each dict has the keys of :func:`split_with_diagnostics` plus ``test_fold_id``, the fold
    tested. ``purged_indices`` lists the earlier samples purging removed; ``embargo_indices`` is
    always empty; samples after the test fold are left out without being listed.
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        _split_dict(d)
        for d in _cv.walk_forward_splits(
            s0,
            s1,
            _count(n_splits, "n_splits"),
            float(pct_embargo),
            _count(min_train_folds, "min_train_folds"),
        )
    ]


def _require_real(value: Any, name: str) -> np.ndarray:
    arr = _unmasked_asarray(value, name)
    if arr.dtype.kind == "c" or (
        arr.dtype.kind == "O"
        and any(isinstance(item, (complex, np.complexfloating)) for item in arr.flat)
    ):
        raise ValueError(f"{name} must be real")
    return arr


def _center_binary64_column(source: np.ndarray) -> np.ndarray:
    """Center finite binary64 values exactly, then round each difference once."""
    # Denominators returned by float.as_integer_ratio are powers of two. Accumulate
    # at the finest precision seen so far, rather than retaining a Fraction per row.
    total = 0
    shift = 0
    for value in source:
        num, den = float(value).as_integer_ratio()
        bits = den.bit_length() - 1
        if bits > shift:
            total <<= bits - shift
            shift = bits
        total += num << (shift - bits)
    size = len(source)
    divisor = size << shift

    def deviations():
        for value in source:
            num, den = float(value).as_integer_ratio()
            exact = (num << (shift - (den.bit_length() - 1))) * size - total
            yield exact / divisor

    return np.fromiter(deviations(), dtype=float, count=size)


def bootstrap_returns(
    returns: Any, demean: bool = True
) -> Callable[[np.random.Generator], np.ndarray]:
    """A no-signal generator for :func:`null_score_distribution`: returns drawn i.i.d.

    Each call draws ``len(returns)`` rows of ``returns`` with replacement (a 1-D array, or a
    2-D array whose rows are drawn whole, keeping the cross-section of each bar). With
    ``demean`` (the default) the mean return is subtracted first, per column, so the drawn
    path is a driftless random walk with the returns' distribution: nothing in it predicts the
    next return. The generator captures a copy of the inputs. Centering that
    cannot be represented as finite float64 raises ValueError. Rebuild prices,
    features **and labels** from it.

    Two tempting alternatives are wrong for features with memory. Permuting the labels alone
    keeps every feature but cuts the link between a label and the path a level feature
    records, so it cannot show the bias of issue #217. Shuffling the returns *without*
    replacement fixes their sum, so every shuffled path ends where the real one does: a
    random-walk bridge, which is mean-reverting, and on which a level feature scores above
    chance even walk-forward.
    """
    arr = _require_real(returns, "returns").astype(float, copy=True)
    if arr.ndim not in (1, 2) or arr.shape[0] < 2:
        raise ValueError("returns must be a 1-D or 2-D array with at least two rows")
    if not np.isfinite(arr).all():
        raise ValueError("returns must be finite")
    if demean:
        # Keep vectorized NumPy results for ordinary columns. A finite mean is
        # not sufficient: cancellation can erase the drift, and dividing a
        # subnormal mean before subtracting can round the wrong deviations.
        # Large common offsets can also absorb the mean correction.
        columns = arr[:, None] if arr.ndim == 1 else arr
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            mean = columns.mean(axis=0)
            centered = arr - mean
            minimum = columns.min(axis=0)
            maximum = columns.max(axis=0)
            scale = np.maximum(np.abs(minimum), np.abs(maximum))
            finfo = np.finfo(float)
            resolution = math.sqrt(finfo.eps) * scale
            sensitive = (scale > 0) & (
                (scale < finfo.tiny)
                | (np.abs(mean) <= resolution)
                | (maximum - minimum <= resolution)
            )
        recovered = centered[:, None] if arr.ndim == 1 else centered
        finite = np.isfinite(recovered).all(axis=0)
        for column in np.flatnonzero(sensitive | ~finite):
            if minimum[column] == maximum[column]:
                recovered[:, column] = 0.0
                continue
            source = columns[:, column]
            if finite[column] and mean[column] == 0.0:
                # A sum of binary64 inputs cannot underflow to zero unless
                # it is zero. Avoid rational arithmetic for zero-drift panels.
                try:
                    if math.fsum(source) == 0.0:
                        continue
                except OverflowError:
                    pass
            try:
                recovered[:, column] = _center_binary64_column(source)
            except OverflowError as exc:
                raise ValueError("demeaned returns must be representable as finite floats") from exc
        arr = centered

    def draw(rng: np.random.Generator) -> np.ndarray:
        return arr[rng.integers(0, arr.shape[0], size=arr.shape[0])]

    return draw


def null_score_distribution(
    evaluate: Callable[[Any, Sequence[Any]], float],
    splits: Sequence[Any],
    make_null: Callable[[np.random.Generator], Any],
    n_null: int = 100,
    seed: int | Sequence[int] | None = 0,
) -> np.ndarray:
    """Scores of a pipeline on ``n_null`` no-signal datasets, all run through the same splits.

    For each draw, ``make_null(rng)`` builds a dataset with no signal (for example
    :func:`bootstrap_returns`, or a simulation of the same length), and
    ``evaluate(data, splits)`` rebuilds features and labels from it, fits on each split's
    training indices, scores on its test indices and returns one number. The splits are fixed,
    so whatever the splitting scheme itself adds to the score (such as purged k-fold's bias
    toward features with memory, issue #217) is in the null too. Compare the real score with
    this distribution (:func:`null_p_value`), not with chance.

    ``make_null`` must return data of the length ``splits`` indexes. ``seed`` seeds one
    ``numpy.random.Generator`` passed to every call, so the distribution is reproducible.
    Returns the ``n_null`` scores in draw order; a ``NaN`` score is kept, not dropped,
    and a masked scalar score is recorded as ``NaN``.
    """
    n_null = _count(n_null, "n_null")
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1, got {n_null}")
    splits = list(splits)
    if not splits:
        raise ValueError("splits cannot be empty")
    rng = np.random.default_rng(seed)
    out = np.empty(n_null)
    for i in range(n_null):
        score: Any = evaluate(make_null(rng), splits)
        if np.ma.isMaskedArray(score) and score.ndim == 0 and np.ma.is_masked(score):
            out[i] = np.nan
        elif isinstance(score, (complex, np.complexfloating)):
            raise ValueError("null score must be real")
        else:
            out[i] = float(score)
    return out


def _score_ratio(value: Any, name: str) -> tuple[int, int] | None:
    """Exact ratio of an integer or floating scalar; None for nonfinite scores."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        if np.ma.is_masked(value):
            return None
        value = value[()]
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be real")
    if isinstance(value, (int, np.integer, np.bool_)):
        return int(value), 1
    ratio = getattr(value, "as_integer_ratio", None)
    if not callable(ratio):
        # Keep legacy float-coercible inputs, but do not swallow conversion errors.
        ratio = float(value).as_integer_ratio
    try:
        return ratio()
    except (OverflowError, ValueError):
        return None


def _float_tail_count(
    scores: np.ndarray, finite: np.ndarray, observed: Any, numerator: int, denominator: int
) -> int:
    def compare_ratios() -> int:
        return sum(
            num * denominator >= numerator * den
            for num, den in (score.as_integer_ratio() for score in scores[finite])
        )

    if isinstance(observed, np.ndarray) and observed.ndim == 0:
        observed = observed[()]
    if not isinstance(observed, (int, float, np.integer, np.floating, np.bool_)):
        # Casting Decimal or Fraction can fall back to binary64 even if the
        # target is longdouble. Compare exact ratios for these uncommon inputs.
        return compare_ratios()
    # Compare in the input dtype against its smallest value >= the exact threshold.
    # Rounding the threshold to nearest instead introduces false ties when it rounds down.
    with np.errstate(over="ignore", under="ignore"):
        try:
            boundary = scores.dtype.type(observed)
        except (OverflowError, ValueError):
            # NumPy's scalar conversion can use a bounded decimal string for
            # huge integer thresholds, even when longdouble's range covers them.
            return compare_ratios()
        if np.isfinite(boundary):
            num, den = boundary.as_integer_ratio()
            if num * denominator < numerator * den:
                boundary = np.nextafter(boundary, scores.dtype.type(math.inf))
    # Reuse the finite mask; do not allocate a compact floating-point copy.
    np.greater_equal(scores, boundary, out=finite, where=finite)
    return int(np.count_nonzero(finite))


def null_p_value(observed: float, null: Any) -> float:
    """One-sided p-value of ``observed`` against a null distribution: P(null >= observed).

    With ``n`` finite null scores of which ``k`` are at least ``observed``, returns
    ``(k + 1) / (n + 1)``, so it is never 0 and a null of 99 draws resolves 0.01. ``NaN``
    and masked null scores are ignored. Tail counts preserve the precision of integer
    and floating-point inputs; precision already lost by callers is not recovered.
    """
    if np.ma.isMaskedArray(null):
        null = null.compressed()
    # Preserve scalars in a mixed sequence before NumPy's common-dtype inference.
    if isinstance(null, np.ndarray):
        original = null
    elif type(null) in (list, tuple) and set(map(type, null)) <= {float}:
        # A homogeneous built-in float sequence is already exact in binary64.
        original = np.asarray(null)
    else:
        original = np.asarray(null, dtype=object)
    arr = _require_real(original, "null scores").ravel()
    if isinstance(observed, (str, bytes)):
        raise TypeError("observed score must be numeric")
    threshold = _score_ratio(observed, "observed score")
    if threshold is None:
        raise ValueError(f"observed must be finite, got {observed}")
    numerator, denominator = threshold
    if arr.dtype.kind in "iu":
        n = arr.size
        ceiling = -(-numerator // denominator)
        bounds = np.iinfo(arr.dtype)
        if ceiling <= bounds.min:
            k = n
        elif ceiling > bounds.max:
            k = 0
        else:
            k = int(np.count_nonzero(arr >= arr.dtype.type(ceiling)))
    elif arr.dtype.kind != "O":
        if arr.dtype.kind != "f":
            arr = arr.astype(float, copy=False)
        finite = np.isfinite(arr)
        n = int(np.count_nonzero(finite))
        k = _float_tail_count(arr, finite, observed, numerator, denominator)
    else:
        n = k = 0
        for value in arr:
            if value is None:
                continue  # NumPy's prior object-to-float conversion maps None to NaN.
            ratio = _score_ratio(value, "null scores")
            if ratio is not None:
                n += 1
                k += ratio[0] * denominator >= numerator * ratio[1]
    if n == 0:
        raise ValueError("null has no finite scores")
    return float((k + 1) / (n + 1))


def naive_kfold_splits(n_samples: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unpurged contiguous k-fold: the leaky baseline AFML §7.3 warns against.

    For measuring leakage with :func:`count_train_test_overlaps`, not for validating models.
    """
    return [
        (_indices(train), _indices(test))
        for train, test in _cv.naive_kfold_splits(
            _count(n_samples, "n_samples"), _count(n_splits, "n_splits")
        )
    ]


def count_train_test_overlaps(
    t0: Any, t1: Any, train_indices: Sequence[int], test_indices: Sequence[int]
) -> int:
    """How many training samples have a label span intersecting some test sample's span."""
    s0, s1 = _label_spans(t0, t1)
    return _cv.count_train_test_overlaps(
        s0,
        s1,
        _index_list(train_indices, "train_indices"),
        _index_list(test_indices, "test_indices"),
    )
