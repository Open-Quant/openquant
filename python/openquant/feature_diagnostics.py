from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, exp, isfinite, lcm, log, log1p, sqrt
from sys import float_info
from typing import Any

import numpy as np
import polars as pl

from . import viz

_EPS = 1e-12


@dataclass(frozen=True)
class _LinearModel:
    coeffs: list[float]
    intercept: float


def _as_matrix(x: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [list(map(float, r)) for r in x]
    if not rows:
        raise ValueError("X cannot be empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("X must contain at least one feature")
    if any(len(r) != width for r in rows):
        raise ValueError("X must be rectangular")
    return rows


def _as_vector(y: Sequence[float], n_rows: int) -> list[float]:
    out = [float(v) for v in y]
    if len(out) != n_rows:
        raise ValueError(f"y/X length mismatch: {len(out)} vs {n_rows}")
    return out


def _feature_names(n_features: int, feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is None:
        return [f"f{i}" for i in range(n_features)]
    out = [str(v) for v in feature_names]
    if len(out) != n_features:
        raise ValueError(f"feature_names length mismatch: expected {n_features}, got {len(out)}")
    return out


def _sample_weight(weights: Sequence[float] | None, n_rows: int) -> list[float] | None:
    if weights is None:
        return None
    out = [float(v) for v in weights]
    if len(out) != n_rows:
        raise ValueError(f"sample_weight/X length mismatch: {len(out)} vs {n_rows}")
    if any(not isfinite(w) or w < 0.0 for w in out):
        raise ValueError("sample_weight must contain finite nonnegative values")
    if not any(w > 0.0 for w in out):
        raise ValueError("sample_weight must contain a positive value")
    return out


def _check_fold_weight_mass(
    weights: Sequence[float] | None,
    splits: Sequence[tuple[list[int], list[int]]],
) -> None:
    if weights is None:
        return
    for fold, (train, test) in enumerate(splits):
        for role, indices in (("train", train), ("test", test)):
            if not any(weights[i] > 0.0 for i in indices):
                raise ValueError(f"sample_weight fold {fold} {role} must contain a positive value")


def _prepare_inputs(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None,
    sample_weight: Sequence[float] | None,
) -> tuple[list[list[float]], list[float], list[str], list[float] | None]:
    """Validated `X`, `y`, feature names and sample weights, in that validation order."""
    x = _as_matrix(X)
    yv = _as_vector(y, len(x))
    names = _feature_names(len(x[0]), feature_names)
    weights = _sample_weight(sample_weight, len(x))
    return x, yv, names, weights


def _cv_splits(
    event_end_indices: Sequence[int] | None,
    n_rows: int,
    n_splits: int,
    pct_embargo: float,
    allow_unpurged: bool,
    weights: Sequence[float] | None,
) -> list[tuple[list[int], list[int]]]:
    """Purged k-fold splits, each checked to keep positive weight on both sides."""
    intervals = _build_intervals(event_end_indices, n_rows, allow_unpurged=allow_unpurged)
    splits = _purged_kfold_splits(intervals, n_splits=n_splits, pct_embargo=pct_embargo)
    _check_fold_weight_mass(weights, splits)
    return splits


def _cv_meta(
    event_end_indices: Sequence[int] | None,
    n_splits: int,
    pct_embargo: float,
    splits: Sequence[tuple[list[int], list[int]]],
    scoring: str,
) -> dict[str, Any]:
    purged = event_end_indices is not None
    return {
        "method": "purged_kfold" if purged else "kfold_embargo_only",
        "purged": purged,
        "n_splits": n_splits,
        "pct_embargo": pct_embargo,
        "fold_count": len(splits),
        "scoring": scoring,
    }


def _fold_data(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    weights: Sequence[float] | None,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> tuple[
    list[Sequence[float]],
    list[float],
    list[Sequence[float]],
    list[float],
    list[float] | None,
    list[float] | None,
]:
    """Training and test rows, labels and weights (None without weights) of one fold."""
    return (
        [x[i] for i in train_idx],
        [y[i] for i in train_idx],
        [x[i] for i in test_idx],
        [y[i] for i in test_idx],
        [weights[i] for i in train_idx] if weights is not None else None,
        [weights[i] for i in test_idx] if weights is not None else None,
    )


def _build_intervals(
    event_end_indices: Sequence[int] | None,
    n_rows: int,
    allow_unpurged: bool = False,
) -> list[tuple[int, int]]:
    if event_end_indices is None:
        if not allow_unpurged:
            raise ValueError(
                "event_end_indices is required for purged cross-validation. Without "
                "label end indices every sample becomes a degenerate one-row interval "
                "(i, i), which can only overlap the test rows themselves, so the purge "
                "step removes nothing and training folds leak label information from "
                "the test fold. Pass event_end_indices (the row index at which each "
                "sample's label is resolved), or pass allow_unpurged=True to accept "
                "embargo-only splits explicitly."
            )
        return [(i, i) for i in range(n_rows)]
    ends = [int(v) for v in event_end_indices]
    if len(ends) != n_rows:
        raise ValueError(f"event_end_indices/X length mismatch: {len(ends)} vs {n_rows}")
    intervals: list[tuple[int, int]] = []
    for i, end in enumerate(ends):
        if end < i:
            raise ValueError("event_end_indices must satisfy end >= start index")
        intervals.append((i, min(end, n_rows - 1)))
    return intervals


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _purged_kfold_splits(
    intervals: Sequence[tuple[int, int]],
    n_splits: int,
    pct_embargo: float,
) -> list[tuple[list[int], list[int]]]:
    n = len(intervals)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > n:
        raise ValueError("n_splits cannot exceed sample count")
    if pct_embargo < 0.0 or pct_embargo >= 1.0:
        raise ValueError("pct_embargo must be in [0.0, 1.0)")

    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1

    splits: list[tuple[list[int], list[int]]] = []
    start = 0
    embargo_n = int(ceil(n * pct_embargo))

    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = list(range(start, stop))
        test_intervals = [intervals[i] for i in test_idx]

        train_mask = [True] * n
        for i, interval in enumerate(intervals):
            if any(_overlaps(interval, t) for t in test_intervals):
                train_mask[i] = False

        # Keep the legacy pre-fold embargo, but count the post-fold embargo
        # from the first row after all test labels resolve, not inside the purge.
        before = max(0, start - embargo_n)
        test_end = max(end for _, end in test_intervals)
        resume = stop
        while resume < n and intervals[resume][0] <= test_end:
            resume += 1
        after = min(n, resume + embargo_n)
        for i in range(before, after):
            train_mask[i] = False

        train_idx = [i for i, keep in enumerate(train_mask) if keep]
        if not train_idx or not test_idx:
            raise ValueError("PurgedKFold generated an empty train/test fold")

        splits.append((train_idx, test_idx))
        start = stop

    return splits


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    ez = exp(z)
    return ez / (1.0 + ez)


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(a)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < _EPS:
            raise ValueError("Singular linear system")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < _EPS:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def _penalized_log_loss(
    design: Sequence[Sequence[float]],
    y: Sequence[float],
    sw: Sequence[float],
    beta: Sequence[float],
    ridge: float,
) -> float:
    loss = 0.0
    for row, yy, w in zip(design, y, sw):
        z = _dot(row, beta)
        # log(1 + e^z) - y z, written to stay finite for large |z|.
        softplus = z + log1p(exp(-z)) if z > 0 else log1p(exp(z))
        loss += w * (softplus - yy * z)
    return loss + 0.5 * ridge * _dot(beta, beta)


def _fraction_logistic(z: Fraction) -> tuple[Fraction, Fraction]:
    """Probability and softplus tail without subtracting a rounded probability from one."""
    try:
        magnitude = float(abs(z))
    except OverflowError:
        magnitude = float("inf")
    tail_exp = exp(-magnitude)
    if tail_exp < float_info.min and abs(z) <= 4096:
        # Eight normal-range factors retain tails below binary64's subnormal
        # floor, with bounded integer sizes (the smallest factor is exp(-512)).
        q = Fraction(exp(-magnitude / 8)) ** 8
        # 0 <= q - log1p(q) <= q²/2: relative error about 1e-308 or less,
        # far below the error from the binary64 exponential factors.
        tail = q
    else:
        q = Fraction(tail_exp)
        tail = Fraction(log1p(tail_exp))
    prob = 1 / (1 + q) if z >= 0 else q / (1 + q)
    return prob, tail


def _fraction_logistic_loss(
    design: list[list[Fraction]],
    y: list[Fraction],
    weights: list[Fraction],
    beta: list[Fraction],
    ridge: Fraction,
) -> Fraction:
    loss = ridge * sum((b * b for b in beta), Fraction()) / 2
    for row, yy, w in zip(design, y, weights):
        if not w:
            continue
        z = sum((v * b for v, b in zip(row, beta)), Fraction())
        _, tail = _fraction_logistic(z)
        loss += w * ((z if z > 0 else Fraction()) - yy * z + tail)
    return loss


def _solve_fraction_system(a: list[list[Fraction]], b: list[Fraction]) -> list[float]:
    """Solve exactly with integer Bareiss elimination, retaining arbitrarily small ridge."""
    # Clearing each row's denominators preserves its equation. Bareiss divisions
    # are exact, avoiding repeated Fraction reduction of large intermediate terms.
    aug = []
    for row, rhs in zip(a, b):
        values = [*row, rhs]
        denominator = lcm(*(v.denominator for v in values))
        aug.append([v.numerator * (denominator // v.denominator) for v in values])
    n = len(b)
    previous = 1
    for col in range(n - 1):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if not aug[pivot][col]:
            raise ValueError("Singular linear system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        current = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col]
            for j in range(col + 1, n + 1):
                aug[r][j] = (current * aug[r][j] - factor * aug[col][j]) // previous
            aug[r][col] = 0
        previous = current
    if n and not aug[-1][-2]:
        raise ValueError("Singular linear system")
    # The final pivot is the integer system determinant (up to row-swap sign).
    # Cramer's rule ensures determinant*x is integral, so back substitution also
    # uses exact integer division. Only final coefficients are rounded to float.
    determinant = aug[-1][-2] if n else 1
    numerators = [0] * n
    for r in range(n - 1, -1, -1):
        top = determinant * aug[r][n] - sum(aug[r][j] * numerators[j] for j in range(r + 1, n))
        numerators[r] = top // aug[r][r]
    return [float(Fraction(v, determinant)) for v in numerators]


def _recover_weighted_logistic(
    design: list[list[float]],
    y: Sequence[float],
    weights: list[float],
    ridge: float,
    max_iter: int,
    tol: float,
) -> _LinearModel | None:
    """Rare finite-input, binary-label recovery preserving the weighted objective and ridge.

    Rational accumulation/elimination prevents overflow and keeps ridge terms on
    rank-deficient designs. Exponential/logarithmic evaluations and returned
    coefficients remain binary64; this is not a general arbitrary-precision fit.
    The original iteration cap remains, so separable extreme-weight problems
    need not reach their ridge optimum within that budget. Exponential tails
    below the normal range use eight-way splitting only for |logit| <= 4096;
    larger logits retain the zero-tail limit. A retry starts from zero with
    the same iteration cap.
    """
    if not (
        isfinite(ridge)
        and ridge > 0
        and all(isfinite(v) for row in design for v in row)
        and all(v in (0.0, 1.0) for v in y)
        and all(isfinite(w) and w >= 0 for w in weights)
        and any(w > 0 for w in weights)
    ):
        return None
    rows = [[Fraction(v) for v in row] for row in design]
    labels = [Fraction(v) for v in y]
    sw = [Fraction(w) for w in weights]
    penalty = Fraction(ridge)
    dim = len(rows[0])
    beta = [0.0] * dim
    objective = _fraction_logistic_loss(rows, labels, sw, [Fraction()] * dim, penalty)
    for _ in range(max_iter):
        bq = [Fraction(b) for b in beta]
        gradient = [penalty * b for b in bq]
        hessian = [[penalty if i == j else Fraction() for j in range(dim)] for i in range(dim)]
        for row, yy, w in zip(rows, labels, sw):
            if not w:
                continue
            z = sum((v * b for v, b in zip(row, bq)), Fraction())
            prob, _ = _fraction_logistic(z)
            resid = w * (prob - yy)
            curv = w * prob * (1 - prob)
            for i in range(dim):
                gradient[i] += resid * row[i]
                for j in range(i, dim):
                    hessian[i][j] += curv * row[i] * row[j]
        for i in range(dim):
            for j in range(i):
                hessian[i][j] = hessian[j][i]
        step = _solve_fraction_system(hessian, gradient)
        scale = 1.0
        while True:
            candidate = [b - scale * s for b, s in zip(beta, step)]
            cand_obj = _fraction_logistic_loss(
                rows, labels, sw, [Fraction(b) for b in candidate], penalty
            )
            if cand_obj <= objective or scale < 1e-10:
                break
            scale *= 0.5
        if cand_obj > objective:
            break
        coefficient_small = max(abs(scale * s) for s in step) < tol * (
            1 + max(abs(b) for b in beta)
        )
        # A tiny coefficient step can still move logits by order one when
        # features are huge. Also retain the coefficient criterion for ridge
        # movement in a likelihood nullspace.
        change = [Fraction(c) - b for c, b in zip(candidate, bq)]
        # Scale each observation by its own logit. A negligible-weight outlier
        # with an enormous logit must not hide progress on the other rows.
        logits_small = all(
            abs(sum((v * d for v, d in zip(row, change)), Fraction()))
            < Fraction(tol) * (1 + abs(sum((v * b for v, b in zip(row, bq)), Fraction())))
            for row, w in zip(rows, sw)
            if w
        )
        converged = coefficient_small and logits_small
        beta, objective = candidate, cand_obj
        if converged:
            break
    return _LinearModel(coeffs=beta[1:], intercept=beta[0])


def _material_loss_cancellation(
    design: list[list[float]],
    y: Sequence[float],
    weights: list[float],
    beta: list[float],
    objective: float,
) -> bool:
    # Called only for weights large relative to the penalty. Tiny harmless tail
    # roundoff must not change the ordinary fitting path.
    for row, yy, w in zip(design, y, weights):
        if yy != 1.0 or w == 0:
            continue
        z = _dot(row, beta)
        if z > 0 and isfinite(z):
            tail = log1p(exp(-z))
            lost = abs(((z + tail) - z) - tail)
            if w * lost > _EPS * abs(objective):
                return True
    return False


def _fit_linear_probability_model(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    sample_weight: Sequence[float] | None,
    ridge: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> _LinearModel:
    """Fit a weighted logistic regression of the 0/1 label by Newton's method (IRLS).

    Despite the historical name this is *not* a linear probability model: least squares
    on a 0/1 label is already on the probability scale, so pairing it with the sigmoid in
    ``_predict_proba`` squeezed every prediction towards 0.5 (#99). The coefficients here
    are on the log-odds scale, which is what the sigmoid expects.

    ``ridge`` is an L2 penalty on all coefficients, intercept included. It keeps the
    Newton system non-singular and the coefficients finite when a fold is perfectly
    separable. Each Newton step is halved until the penalised log loss decreases, so the
    iteration cannot diverge for finite arithmetic. Nonfinite arithmetic or a
    singular floating-point solve on finite inputs with binary labels and positive ridge retries
    using rational accumulation/elimination. Weights above max(1, ridge)/1e-12
    also check for material loss-tail cancellation before accepting a step.
    Recovery retains the iteration cap and requires small coefficient and logit
    changes; its exponential/logarithmic evaluations still use binary64.
    """
    n = len(x)
    p = len(x[0])
    sw = [1.0] * n if sample_weight is None else [float(v) for v in sample_weight]

    dim = p + 1
    design = [[1.0] + list(row) for row in x]
    beta = [0.0] * dim
    objective = _penalized_log_loss(design, y, sw, beta, ridge)
    extreme_weights = max(sw) > max(1.0, ridge) / _EPS
    if not isfinite(objective):
        recovered = _recover_weighted_logistic(design, y, sw, ridge, max_iter, tol)
        if recovered is not None:
            return recovered

    for _ in range(max_iter):
        hessian = [[0.0 for _ in range(dim)] for _ in range(dim)]
        gradient = [ridge * b for b in beta]
        for row, yy, w in zip(design, y, sw):
            prob = _sigmoid(_dot(row, beta))
            resid = w * (prob - yy)
            curv = w * prob * (1.0 - prob)
            for i in range(dim):
                gradient[i] += resid * row[i]
                ci = curv * row[i]
                for j in range(i, dim):
                    hessian[i][j] += ci * row[j]
        for i in range(dim):
            hessian[i][i] += ridge
            for j in range(i):
                hessian[i][j] = hessian[j][i]

        if any(not isfinite(v) for row in hessian for v in row) or any(
            not isfinite(v) for v in gradient
        ):
            recovered = _recover_weighted_logistic(design, y, sw, ridge, max_iter, tol)
            if recovered is not None:
                return recovered
        try:
            step = _solve_linear_system(hessian, gradient)
        except ValueError:
            recovered = _recover_weighted_logistic(design, y, sw, ridge, max_iter, tol)
            if recovered is not None:
                return recovered
            raise
        scale = 1.0
        while True:
            candidate = [b - scale * s for b, s in zip(beta, step)]
            cand_obj = _penalized_log_loss(design, y, sw, candidate, ridge)
            if not isfinite(cand_obj) or (
                extreme_weights and _material_loss_cancellation(design, y, sw, candidate, cand_obj)
            ):
                recovered = _recover_weighted_logistic(design, y, sw, ridge, max_iter, tol)
                if recovered is not None:
                    return recovered
            if cand_obj <= objective or scale < 1e-10:
                break
            scale *= 0.5
        if cand_obj > objective:
            break
        converged = max(abs(scale * s) for s in step) < tol * (1.0 + max(abs(b) for b in beta))
        beta, objective = candidate, cand_obj
        if converged:
            break

    return _LinearModel(coeffs=beta[1:], intercept=beta[0])


def _predict_proba(model: _LinearModel, x: Sequence[Sequence[float]]) -> list[float]:
    """Recover nonfinite or cancellation-sensitive evaluation of the stored model.

    Estimated roundoff above 1e-10 * max(1, abs(logit)) selects exceptional
    finite dot products. This is a diagnostic policy, not a universal bound
    on prediction error.
    """
    unit_error = (len(model.coeffs) + 2) * (float_info.epsilon / 2)
    gamma = unit_error / (1 - unit_error)
    probabilities = []
    for row in x:
        products = [v * b for v, b in zip(row, model.coeffs)]
        logit = model.intercept + sum(products)
        magnitude = abs(model.intercept) + sum(map(abs, products))
        recover = not isfinite(logit) or gamma * magnitude > 1e-10 * max(1.0, abs(logit))
        if (
            recover
            and isfinite(model.intercept)
            and all(isfinite(v) for v in model.coeffs)
            and all(isfinite(v) for v in row)
        ):
            # Recover both product and summation error. This evaluates the
            # stored coefficients, not an unrepresentable ideal optimum.
            exact = Fraction(model.intercept) + sum(
                (Fraction(v) * Fraction(b) for v, b in zip(row, model.coeffs)), Fraction()
            )
            try:
                logit = float(exact)
            except OverflowError:
                logit = float("inf") if exact > 0 else -float("inf")
        probabilities.append(_sigmoid(logit))
    return probabilities


def _exact_weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    # Exceptional scoring only: preserve the supplied finite weights and the
    # binary64 values being averaged, without rounding intermediate products.
    exact_weights = [Fraction(w) for w in weights]
    numerator = sum((w * Fraction(v) for w, v in zip(exact_weights, values)), Fraction())
    return float(numerator / sum(exact_weights))


def _score(
    y_true: Sequence[float],
    prob: Sequence[float],
    scoring: str,
    sample_weight: Sequence[float] | None,
) -> float:
    """Score with lazy exact weighted sums for exceptional finite weight arithmetic.

    Ordinary floating-point evaluation is retained. Overflowed sums, log-loss
    weight totals below 1e-150, and subnormal F1 precision-recall products use
    rational arithmetic. Log loss averages the same rounded per-row logarithms;
    this is not an exact logarithm implementation. Model fitting is unaffected.
    """
    weights = [1.0] * len(y_true) if sample_weight is None else [float(v) for v in sample_weight]
    # Validate without summing: finite positive weights can have an infinite sum.
    _sample_weight(weights, len(y_true))
    den = sum(weights)

    if scoring == "neg_log_loss":
        loss = 0.0
        for y, p, w in zip(y_true, prob, weights):
            p_clip = min(max(p, 1e-15), 1.0 - 1e-15)
            loss += w * (-(y * log(p_clip) + (1.0 - y) * log(1.0 - p_clip)))
        if not isfinite(loss) or not isfinite(den) or den < 1e-150:
            losses = []
            for y, p in zip(y_true, prob):
                p_clip = min(max(p, 1e-15), 1.0 - 1e-15)
                losses.append(-(y * log(p_clip) + (1.0 - y) * log(1.0 - p_clip)))
            # Nonfinite row losses remain outside weight-arithmetic recovery.
            if all(isfinite(v) for v in losses):
                return -_exact_weighted_mean(losses, weights)
        return -(loss / den)

    pred = [1.0 if p >= 0.5 else 0.0 for p in prob]

    if scoring == "accuracy":
        correct = sum(w for y, p, w in zip(y_true, pred, weights) if abs(y - p) < 1e-12)
        if not isfinite(den):
            return _exact_weighted_mean(
                [float(abs(y - p) < 1e-12) for y, p in zip(y_true, pred)], weights
            )
        return correct / den

    if scoring == "f1":
        tp = sum(w for y, p, w in zip(y_true, pred, weights) if y > 0.5 and p > 0.5)
        fp = sum(w for y, p, w in zip(y_true, pred, weights) if y <= 0.5 and p > 0.5)
        fn = sum(w for y, p, w in zip(y_true, pred, weights) if y > 0.5 and p <= 0.5)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        numerator = 2.0 * precision * recall
        if not isfinite(tp + fp + fn) or (tp > 0.0 and numerator < float.fromhex("0x1p-1022")):
            # True negatives do not contribute to F1, even when their weights
            # dwarf the relevant positive counts. Avoid precision*recall underflow.
            exact_tp = Fraction()
            exact_fp = Fraction()
            exact_fn = Fraction()
            for y, p, w in zip(y_true, pred, weights):
                if y > 0.5 and p > 0.5:
                    exact_tp += Fraction(w)
                elif y <= 0.5 and p > 0.5:
                    exact_fp += Fraction(w)
                elif y > 0.5 and p <= 0.5:
                    exact_fn += Fraction(w)
            total = 2 * exact_tp + exact_fp + exact_fn
            return float(2 * exact_tp / total) if total else 0.0
        return numerator / (precision + recall) if precision + recall > 0 else 0.0

    raise ValueError("scoring must be one of: neg_log_loss, accuracy, f1")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return sqrt(var)


def _importance_table(
    feature_names: Sequence[str], per_feature_values: Sequence[Sequence[float]]
) -> pl.DataFrame:
    rows = []
    for name, vals in zip(feature_names, per_feature_values):
        rows.append(
            {
                "feature": name,
                "mean": _mean(vals),
                "std": _std(vals),
                "stderr": _std(vals) / sqrt(len(vals)) if vals else 0.0,
            }
        )
    return pl.DataFrame(rows).sort("mean", descending=True)


def _importance_result(
    method: str, feature_names: Sequence[str], per_feature_values: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """`method`, the importance `table`, its `records` and its bar-chart `viz_payload`."""
    table = _importance_table(feature_names, per_feature_values)
    payload = viz.prepare_feature_importance_payload(
        table["feature"].to_list(), table["mean"].to_list(), std=table["stderr"].to_list()
    )
    return {"method": method, "table": table, "records": table.to_dicts(), "viz_payload": payload}


def mdi_importance(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
    n_estimators: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate feature importance from bootstrapped logistic-regression coefficients.

    Named after mean decrease impurity (AFML section 8.3.1), but no trees are grown: for
    each of `n_estimators` bootstrap resamples, an L2-penalized (ridge 1e-3) logistic
    regression is fitted by Newton's method and the absolute coefficients are normalized to
    sum to 1. The importance is the mean of those shares across resamples. It is in-sample,
    and the coefficients are on the raw feature scale, so standardize features first or
    larger-scaled features get smaller shares.

    Parameters
    ----------
    X : Sequence[Sequence[float]]
        Feature matrix, one row per sample; must be non-empty and rectangular.
    y : Sequence[float]
        Binary label per sample, 0.0 or 1.0.
    feature_names : Sequence[str] or None, default None
        Feature names; defaults to `f0`, `f1`, ...
    sample_weight : Sequence[float] or None, default None
        Optional finite nonnegative weight per sample, with at least one positive value.
        Each bootstrap training draw must contain positive weight; a draw with only
        zero-weight rows raises ValueError. Zero-weight rows are otherwise allowed.
    n_estimators : int, default 32
        Number of bootstrap resamples; at least 2.
    seed : int, default 42
        Seed for the bootstrap draws.

    Returns
    -------
    dict[str, Any]
        `method` (`"mdi"`), `table` (polars.DataFrame with columns `feature`, `mean`, `std`
        and `stderr` across resamples, sorted by `mean` descending), `records` (`table` as a
        list of dicts), `viz_payload` (`openquant.viz.prepare_feature_importance_payload` of
        `mean` with `stderr` error bars) and `meta` (`n_estimators`, `seed`).

    Raises
    ------
    ValueError
        If `X` is empty or not rectangular, a length does not match the rows of `X`, or
        `n_estimators < 2`.
    """
    x, yv, names, weights = _prepare_inputs(X, y, feature_names, sample_weight)

    if n_estimators < 2:
        raise ValueError("n_estimators must be >= 2")

    rng = random.Random(seed)
    per_feature: list[list[float]] = [[] for _ in names]

    for _ in range(n_estimators):
        idx = [rng.randrange(len(x)) for _ in range(len(x))]
        xb = [x[i] for i in idx]
        yb = [yv[i] for i in idx]
        wb = [weights[i] for i in idx] if weights is not None else None
        if wb is not None and not any(w > 0.0 for w in wb):
            raise ValueError("sample_weight bootstrap training draw must contain a positive value")

        model = _fit_linear_probability_model(xb, yb, wb)
        raw = [abs(v) for v in model.coeffs]
        denom = sum(raw)
        norm = [v / denom if denom > 0 else 0.0 for v in raw]
        for j, v in enumerate(norm):
            per_feature[j].append(v)

    return {
        **_importance_result("mdi", names, per_feature),
        "meta": {"n_estimators": n_estimators, "seed": seed},
    }


def _score_with_perm_groups(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    x_test: Sequence[Sequence[float]],
    y_test: Sequence[float],
    groups: Sequence[Sequence[int]],
    scoring: str,
    sample_weight_train: Sequence[float] | None,
    sample_weight_test: Sequence[float] | None,
    rng: random.Random,
) -> tuple[float, list[float]]:
    model = _fit_linear_probability_model(x_train, y_train, sample_weight_train)
    base = _score(y_test, _predict_proba(model, x_test), scoring, sample_weight_test)

    out: list[float] = []
    n = len(x_test)
    for cols in groups:
        # Shuffle the rows of the group's columns, as AFML Snippet 8.3 does with
        # np.random.shuffle. All columns of a group get the same row order, so their
        # joint distribution is kept and only their link to the label is broken.
        order = list(range(n))
        rng.shuffle(order)
        perm = [list(row) for row in x_test]
        for c in cols:
            for i in range(n):
                perm[i][c] = x_test[order[i]][c]

        perm_score = _score(y_test, _predict_proba(model, perm), scoring, sample_weight_test)
        if scoring == "neg_log_loss":
            imp = (base - perm_score) / (-perm_score) if abs(perm_score) > _EPS else 0.0
        else:
            imp = (base - perm_score) / (1.0 - perm_score) if abs(1.0 - perm_score) > _EPS else 0.0
        out.append(imp if isfinite(imp) else 0.0)

    return base, out


def mda_importance(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
    event_end_indices: Sequence[int] | None = None,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    allow_unpurged: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate out-of-sample permutation feature importance (MDA, AFML section 8.3.2).

    For each fold a model is fitted on the training rows and scored on the test rows; then
    each feature's test column is shuffled once (seeded) and the test rows rescored. As in
    AFML Snippet 8.3, a feature's fold importance is `(base - permuted) / -permuted` for
    `neg_log_loss` and `(base - permuted) / (1 - permuted)` for `accuracy` and `f1`;
    non-finite values become 0.0.

    Cross-validation is purged k-fold (AFML section 7.4): contiguous test folds, training
    samples whose `[i, event_end_indices[i]]` span overlaps any test span are removed, and
    `ceil(n * pct_embargo)` rows before the test fold and after its latest label end are
    also removed. The post-fold embargo starts at the first row whose start is strictly
    after that end. The model is an L2-penalized (ridge 1e-3) logistic regression fitted
    by Newton's method, not the tree ensemble AFML uses.

    Parameters
    ----------
    X : Sequence[Sequence[float]]
        Feature matrix, one row per sample; must be non-empty and rectangular.
    y : Sequence[float]
        Binary label per sample, 0.0 or 1.0.
    feature_names : Sequence[str] or None, default None
        Feature names; defaults to `f0`, `f1`, ...
    sample_weight : Sequence[float] or None, default None
        Optional finite nonnegative weight per sample, with at least one positive value.
        Cross-validation requires positive weight in each training and test fold.
        Zero-weight rows are allowed.
    event_end_indices : Sequence[int] or None, default None
        Row index at which each sample's label is resolved (must be `>=` the sample's own
        index; values past the last row are clipped). Required unless `allow_unpurged`.
    n_splits : int, default 5
        Number of contiguous folds (2 to the number of samples).
    pct_embargo : float, default 0.01
        Fraction of samples, rounded up, embargoed before each test fold and after its
        latest label end; in `[0, 1)`. The pre-fold window is counted from the first
        test row; the post-fold window starts strictly after all test labels resolve.
    scoring : {"neg_log_loss", "accuracy", "f1"}, default "neg_log_loss"
        Out-of-sample score; accuracy and F1 threshold the probability at 0.5.
    allow_unpurged : bool, default False
        Accept `event_end_indices=None`, giving embargo-only splits that leak label
        information across folds.
    seed : int, default 42
        Seed for the permutations.

    Returns
    -------
    dict[str, Any]
        `method` (`"mda"`),
        `table` (polars.DataFrame with columns `feature`, `mean`, `std` and `stderr`
        across folds, sorted by `mean` descending), `records` (`table` as a list of dicts),
        `viz_payload` (`openquant.viz.prepare_feature_importance_payload` of `mean` with
        `stderr` error bars) and `cv`
        (`method` `"purged_kfold"` or `"kfold_embargo_only"`, `purged`, `n_splits`,
        `pct_embargo`, `fold_count`, `scoring`, `seed` and `mean_base_score`, the
        unpermuted score averaged over folds).

    Raises
    ------
    ValueError
        If `X` is empty or not rectangular, a length does not match the rows of `X`,
        `event_end_indices` is None without `allow_unpurged` or has an entry below its row
        index, `n_splits` or `pct_embargo` is out of range, a fold ends up with an empty
        training set, or `scoring` is not supported.
    """
    x, yv, names, weights = _prepare_inputs(X, y, feature_names, sample_weight)
    splits = _cv_splits(event_end_indices, len(x), n_splits, pct_embargo, allow_unpurged, weights)

    per_feature: list[list[float]] = [[] for _ in names]
    fold_scores: list[float] = []
    rng = random.Random(seed)

    for train_idx, test_idx in splits:
        x_train, y_train, x_test, y_test, w_train, w_test = _fold_data(
            x, yv, weights, train_idx, test_idx
        )
        base, scores = _score_with_perm_groups(
            x_train,
            y_train,
            x_test,
            y_test,
            [[j] for j in range(len(names))],
            scoring,
            w_train,
            w_test,
            rng,
        )
        fold_scores.append(base)
        for j, imp in enumerate(scores):
            per_feature[j].append(imp)

    return {
        **_importance_result("mda", names, per_feature),
        "cv": {
            **_cv_meta(event_end_indices, n_splits, pct_embargo, splits, scoring),
            "seed": seed,
            "mean_base_score": _mean(fold_scores),
        },
    }


def sfi_importance(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
    event_end_indices: Sequence[int] | None = None,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    allow_unpurged: bool = False,
) -> dict[str, Any]:
    """Score each feature on its own out of sample (single feature importance, AFML 8.4.1).

    For every feature and fold, a model using only that feature is fitted on the training
    rows and scored on the test rows (AFML Snippet 8.4). The importance is the raw score, so
    with `neg_log_loss` it is negative and higher is better. Being one feature at a time, it
    does not suffer MDA's substitution effect but ignores joint effects.

    Cross-validation is purged k-fold (AFML section 7.4): contiguous test folds, training
    samples whose `[i, event_end_indices[i]]` span overlaps any test span are removed, and
    `ceil(n * pct_embargo)` rows before the test fold and after its latest label end are
    also removed. The post-fold embargo starts at the first row whose start is strictly
    after that end. The model is an L2-penalized (ridge 1e-3) logistic regression fitted
    by Newton's method, not the tree ensemble AFML uses.

    Parameters
    ----------
    X : Sequence[Sequence[float]]
        Feature matrix, one row per sample; must be non-empty and rectangular.
    y : Sequence[float]
        Binary label per sample, 0.0 or 1.0.
    feature_names : Sequence[str] or None, default None
        Feature names; defaults to `f0`, `f1`, ...
    sample_weight : Sequence[float] or None, default None
        Optional finite nonnegative weight per sample, with at least one positive value.
        Cross-validation requires positive weight in each training and test fold.
        Zero-weight rows are allowed.
    event_end_indices : Sequence[int] or None, default None
        Row index at which each sample's label is resolved (must be `>=` the sample's own
        index; values past the last row are clipped). Required unless `allow_unpurged`.
    n_splits : int, default 5
        Number of contiguous folds (2 to the number of samples).
    pct_embargo : float, default 0.01
        Fraction of samples, rounded up, embargoed before each test fold and after its
        latest label end; in `[0, 1)`. The pre-fold window is counted from the first
        test row; the post-fold window starts strictly after all test labels resolve.
    scoring : {"neg_log_loss", "accuracy", "f1"}, default "neg_log_loss"
        Out-of-sample score; accuracy and F1 threshold the probability at 0.5.
    allow_unpurged : bool, default False
        Accept `event_end_indices=None`, giving embargo-only splits that leak label
        information across folds.

    Returns
    -------
    dict[str, Any]
        `method` (`"sfi"`),
        `table` (polars.DataFrame with columns `feature`, `mean`, `std` and `stderr`
        across folds, sorted by `mean` descending), `records` (`table` as a list of dicts),
        `viz_payload` (`openquant.viz.prepare_feature_importance_payload` of `mean` with
        `stderr` error bars) and `cv`
        (`method` `"purged_kfold"` or `"kfold_embargo_only"`, `purged`, `n_splits`,
        `pct_embargo`, `fold_count` and `scoring`).

    Raises
    ------
    ValueError
        If `X` is empty or not rectangular, a length does not match the rows of `X`,
        `event_end_indices` is None without `allow_unpurged` or has an entry below its row
        index, `n_splits` or `pct_embargo` is out of range, a fold ends up with an empty
        training set, or `scoring` is not supported.
    """
    x, yv, names, weights = _prepare_inputs(X, y, feature_names, sample_weight)
    splits = _cv_splits(event_end_indices, len(x), n_splits, pct_embargo, allow_unpurged, weights)

    per_feature: list[list[float]] = [[] for _ in names]

    for j in range(len(names)):
        xj = [[row[j]] for row in x]
        for train_idx, test_idx in splits:
            x_train, y_train, x_test, y_test, w_train, w_test = _fold_data(
                xj, yv, weights, train_idx, test_idx
            )
            model = _fit_linear_probability_model(x_train, y_train, w_train)
            score_val = _score(y_test, _predict_proba(model, x_test), scoring, w_test)
            per_feature[j].append(score_val)

    return {
        **_importance_result("sfi", names, per_feature),
        "cv": _cv_meta(event_end_indices, n_splits, pct_embargo, splits, scoring),
    }


def _scaled_gram(z: Sequence[Sequence[float]]) -> list[list[float]]:
    """`z'z / (n - 1)`: the correlation matrix of columns standardized by `_standardize`."""
    n = len(z)
    p = len(z[0])
    out = [[0.0 for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(p):
            out[i][j] = sum(row[i] * row[j] for row in z) / max(n - 1, 1)
    return out


def _standardize(
    x: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    n = len(x)
    p = len(x[0])
    means = [sum(row[j] for row in x) / n for j in range(p)]
    stds = []
    for j in range(p):
        var = sum((row[j] - means[j]) ** 2 for row in x) / max(n - 1, 1)
        stds.append(sqrt(var) if var > 0 else 1.0)

    z = [[(row[j] - means[j]) / stds[j] for j in range(p)] for row in x]
    return z, means, stds


def orthogonalize_features_pca(
    X: Sequence[Sequence[float]],
    variance_threshold: float = 0.95,
) -> dict[str, Any]:
    """Project standardized features onto their leading principal components (AFML 8.4.2).

    Each column is standardized with its mean and sample standard deviation (a constant
    column is divided by 1). A symmetric eigensolver finds all eigenpairs of the resulting
    correlation matrix; eigenvalues `<= 1e-12` are discarded. The
    fewest leading components whose explained-variance ratios reach `variance_threshold`
    are kept. Component signs are arbitrary.

    Parameters
    ----------
    X : Sequence[Sequence[float]]
        Feature matrix, one row per sample; must be non-empty and rectangular.
    variance_threshold : float, default 0.95
        Cumulative explained-variance ratio to reach, in `(0, 1]`.

    Returns
    -------
    dict[str, Any]
        `transformed` (list of rows of component scores), `table` (polars.DataFrame of the
        scores with columns `pc1`, `pc2`, ...), `records` (`table` as a list of dicts),
        `components` (kept eigenvectors, one list per component, in feature order),
        `explained_variance_ratio` (per kept component), `mean` and `std` (per feature, as
        used for standardizing).

    Raises
    ------
    ValueError
        If `variance_threshold` is not finite and in `(0, 1]`, `X` is empty, not
        rectangular or contains a non-finite value, or every eigenvalue is zero.
    """
    if not isfinite(variance_threshold) or variance_threshold <= 0.0 or variance_threshold > 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")

    x = _as_matrix(X)
    if not np.isfinite(x).all():
        raise ValueError("X must contain only finite values")
    z, means, stds = _standardize(x)
    cov = _scaled_gram(z)

    # A fixed power-iteration seed can be orthogonal to a nonzero eigenspace (for
    # example, the all-ones vector for perfectly anticorrelated columns). Reusing
    # that seed after deflation also loses repeated eigenvalues. `eigh` returns a
    # complete orthonormal basis for this real symmetric matrix, in ascending order.
    values, vectors = np.linalg.eigh(cov)
    order = np.flatnonzero(values > _EPS)[::-1]
    eigvals = values[order].tolist()
    eigvecs = vectors[:, order].T.tolist()

    total = sum(max(v, 0.0) for v in eigvals)
    if total <= _EPS:
        raise ValueError("PCA failed: zero variance matrix")

    kept = 0
    running = 0.0
    explained_ratio: list[float] = []
    for eig in eigvals:
        r = max(eig, 0.0) / total
        explained_ratio.append(r)
        running += r
        kept += 1
        if running >= variance_threshold:
            break

    eigvecs = eigvecs[:kept]
    explained_ratio = explained_ratio[:kept]

    transformed = []
    for row in z:
        transformed.append([_dot(row, comp) for comp in eigvecs])

    columns = {f"pc{i + 1}": [row[i] for row in transformed] for i in range(kept)}
    table = pl.DataFrame(columns)

    return {
        "transformed": transformed,
        "table": table,
        "records": table.to_dicts(),
        "components": eigvecs,
        "explained_variance_ratio": explained_ratio,
        "mean": means,
        "std": stds,
    }


def _corr_matrix(x: Sequence[Sequence[float]]) -> list[list[float]]:
    z, _, _ = _standardize(x)
    return _scaled_gram(z)


def _max_abs_offdiag(m: Sequence[Sequence[float]]) -> float:
    n = len(m)
    if n <= 1:
        return 0.0
    vals = [abs(m[i][j]) for i in range(n) for j in range(n) if i != j]
    return max(vals) if vals else 0.0


def substitution_effect_report(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
    event_end_indices: Sequence[int] | None = None,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    corr_threshold: float = 0.9,
    orthogonalize: bool = True,
    allow_unpurged: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Flag correlated feature pairs whose MDA importance is diluted by substitution.

    AFML section 8.2: when two features carry the same information, permuting either alone
    barely hurts the model, so MDA understates both. This runs `mda_importance` as a
    baseline, then for every pair with `abs(corr) >= corr_threshold` (Pearson correlation
    over all rows) permutes the two columns jointly in each fold. `dilution_ratio` is the
    pair's joint importance over the sum of their baseline MDA means, and a ratio above
    1.15 flags substitution risk. With `orthogonalize=True` it also reruns MDA on the
    principal components (`orthogonalize_features_pca` at a fixed 0.95 variance threshold),
    which AFML recommends as a remedy.

    Parameters
    ----------
    X : Sequence[Sequence[float]]
        Feature matrix, one row per sample; must be non-empty and rectangular.
    y : Sequence[float]
        Binary label per sample, 0.0 or 1.0.
    feature_names : Sequence[str] or None, default None
        Feature names; defaults to `f0`, `f1`, ...
    sample_weight : Sequence[float] or None, default None
        Optional finite nonnegative weight per sample, with at least one positive value.
        Cross-validation requires positive weight in each training and test fold.
        Zero-weight rows are allowed.
    event_end_indices : Sequence[int] or None, default None
        Row index at which each sample's label is resolved (must be `>=` the sample's own
        index; values past the last row are clipped). Required unless `allow_unpurged`.
    n_splits : int, default 5
        Number of contiguous folds (2 to the number of samples).
    pct_embargo : float, default 0.01
        Fraction of samples, rounded up, embargoed before each test fold and after its
        latest label end; in `[0, 1)`. The pre-fold window is counted from the first
        test row; the post-fold window starts strictly after all test labels resolve.
    scoring : {"neg_log_loss", "accuracy", "f1"}, default "neg_log_loss"
        Out-of-sample score; accuracy and F1 threshold the probability at 0.5.
    corr_threshold : float, default 0.9
        Minimum absolute correlation for a pair to be tested.
    orthogonalize : bool, default True
        Also compute MDA on principal components.
    allow_unpurged : bool, default False
        Accept `event_end_indices=None`, giving embargo-only splits that leak label
        information across folds.
    seed : int, default 42
        Seed for the permutations.

    Returns
    -------
    dict[str, Any]
        `baseline_mda` (the `mda_importance` result), `pairs` (polars.DataFrame with columns
        `feature_a`, `feature_b`, `corr`, `single_sum`, `group_importance`,
        `dilution_ratio` and `flag_substitution_risk`, one row per tested pair) and
        `pair_records` (`pairs` as a list of dicts). With `orthogonalize=True`, also
        `orthogonalized` (`pca`: the `orthogonalize_features_pca` result, `mda`: MDA on the
        components, `max_abs_corr_before` and `max_abs_corr_after`: largest absolute
        off-diagonal correlation of the raw features and of the components) and
        `comparison_viz_payload` (grouped bars of raw against orthogonalized MDA means).

    Raises
    ------
    ValueError
        If `X` is empty or not rectangular, a length does not match the rows of `X`,
        `event_end_indices` is None without `allow_unpurged` or has an entry below its row
        index, `n_splits` or `pct_embargo` is out of range, a fold ends up with an empty
        training set, or `scoring` is not supported; also if `orthogonalize` is True and
        every feature is constant.
    """
    x, yv, names, weights = _prepare_inputs(X, y, feature_names, sample_weight)
    splits = _cv_splits(event_end_indices, len(x), n_splits, pct_embargo, allow_unpurged, weights)

    mda = mda_importance(
        x,
        yv,
        feature_names=names,
        sample_weight=weights,
        event_end_indices=event_end_indices,
        n_splits=n_splits,
        pct_embargo=pct_embargo,
        scoring=scoring,
        allow_unpurged=allow_unpurged,
        seed=seed,
    )
    base_table: pl.DataFrame = mda["table"]
    base_map = {row["feature"]: float(row["mean"]) for row in base_table.to_dicts()}

    corr = _corr_matrix(x)
    pairs: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr_ij = corr[i][j]
            if abs(corr_ij) < corr_threshold:
                continue

            grouped_vals: list[float] = []
            for train_idx, test_idx in splits:
                x_train, y_train, x_test, y_test, w_train, w_test = _fold_data(
                    x, yv, weights, train_idx, test_idx
                )
                _, group_imp = _score_with_perm_groups(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    [[i, j]],
                    scoring,
                    w_train,
                    w_test,
                    rng,
                )
                grouped_vals.append(group_imp[0])

            group_mean = _mean(grouped_vals)
            single_sum = base_map[names[i]] + base_map[names[j]]
            dilution_ratio = group_mean / (single_sum + _EPS)
            pairs.append(
                {
                    "feature_a": names[i],
                    "feature_b": names[j],
                    "corr": corr_ij,
                    "single_sum": single_sum,
                    "group_importance": group_mean,
                    "dilution_ratio": dilution_ratio,
                    "flag_substitution_risk": dilution_ratio > 1.15,
                }
            )

    pairs_df = (
        pl.DataFrame(pairs)
        if pairs
        else pl.DataFrame(
            {
                "feature_a": [],
                "feature_b": [],
                "corr": [],
                "single_sum": [],
                "group_importance": [],
                "dilution_ratio": [],
                "flag_substitution_risk": [],
            }
        )
    )

    out: dict[str, Any] = {
        "baseline_mda": mda,
        "pairs": pairs_df,
        "pair_records": pairs_df.to_dicts(),
    }

    if orthogonalize:
        ortho = orthogonalize_features_pca(x, variance_threshold=0.95)
        x_ortho = ortho["transformed"]
        pc_names = [f"pc{i + 1}" for i in range(len(x_ortho[0]))]
        ortho_mda = mda_importance(
            x_ortho,
            yv,
            feature_names=pc_names,
            sample_weight=weights,
            event_end_indices=event_end_indices,
            n_splits=n_splits,
            pct_embargo=pct_embargo,
            scoring=scoring,
            allow_unpurged=allow_unpurged,
            seed=seed,
        )
        corr_abs_max = _max_abs_offdiag(corr)
        ortho_corr = _corr_matrix(x_ortho)
        ortho_corr_abs_max = _max_abs_offdiag(ortho_corr)
        out["orthogonalized"] = {
            "pca": ortho,
            "mda": ortho_mda,
            "max_abs_corr_before": corr_abs_max,
            "max_abs_corr_after": ortho_corr_abs_max,
        }

        out["comparison_viz_payload"] = viz.prepare_feature_importance_comparison_payload(
            left_labels=base_table["feature"].to_list(),
            left_values=base_table["mean"].to_list(),
            right_labels=ortho_mda["table"]["feature"].to_list(),
            right_values=ortho_mda["table"]["mean"].to_list(),
            left_name="mda_raw",
            right_name="mda_orthogonalized",
        )

    return out


def _screen_rows(
    X: Sequence[Sequence[float]] | pl.DataFrame, feature_names: Sequence[str] | None
) -> tuple[Sequence[Sequence[Any]], list[str]]:
    """`feature_screen_report`'s rows (None and non-finite values kept) and feature names."""
    if isinstance(X, pl.DataFrame):
        if X.width == 0 or X.height == 0:
            raise ValueError("X cannot be empty")
        names = list(feature_names) if feature_names is not None else [str(c) for c in X.columns]
        if feature_names is not None and len(names) != X.width:
            raise ValueError(f"feature_names length mismatch: expected {X.width}, got {len(names)}")
        return X.select([pl.col(c) for c in X.columns]).rows(), names
    rows = [list(r) for r in X]
    if not rows:
        raise ValueError("X cannot be empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("X must contain at least one feature")
    if any(len(r) != width for r in rows):
        raise ValueError("X must be rectangular")
    return rows, _feature_names(width, feature_names)


def _pairwise_corr(rows: Sequence[Sequence[Any]], i: int, j: int) -> float:
    """Pearson correlation of columns `i` and `j` over the rows where both are finite.

    0.0 with fewer than two such rows or when either column is (near) constant on them.
    """
    pairs: list[tuple[float, float]] = []
    for row in rows:
        vi = row[i]
        vj = row[j]
        if vi is None or vj is None:
            continue
        fi = float(vi)
        fj = float(vj)
        if isfinite(fi) and isfinite(fj):
            pairs.append((fi, fj))
    if len(pairs) < 2:
        return 0.0
    ai = [p[0] for p in pairs]
    aj = [p[1] for p in pairs]
    sdi = _std(ai)
    sdj = _std(aj)
    if sdi <= _EPS or sdj <= _EPS:
        return 0.0
    mi = _mean(ai)
    mj = _mean(aj)
    cov = sum((u - mi) * (v - mj) for u, v in pairs) / (len(pairs) - 1)
    return cov / (sdi * sdj)


def feature_screen_report(
    X: Sequence[Sequence[float]] | pl.DataFrame,
    y: Sequence[float] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    min_coverage: float = 0.95,
    max_corr: float = 0.95,
) -> dict[str, Any]:
    """Run lightweight feature QA checks for notebook discovery loops.

    Per feature, `coverage` is the share of rows holding a finite value (None and
    non-finite values count as missing) and `std` the sample standard deviation of those
    values. A feature is rejected for coverage below `min_coverage`, for a standard
    deviation at most 1e-12, or as the weaker of a pair whose absolute Pearson correlation
    (over the rows where both are finite) exceeds `max_corr`. Pairs are resolved greedily,
    strongest correlation first, keeping the feature with the higher `(coverage, std)`;
    a pair with an already dropped feature is skipped.

    Parameters
    ----------
    X : Sequence[Sequence[float]] or polars.DataFrame
        Feature matrix, one row per sample; must be non-empty and rectangular. A frame's
        columns are the features.
    y : Sequence[float] or None, default None
        Optional labels; only their length is checked against `X`.
    feature_names : Sequence[str] or None, default None
        Feature names; defaults to the frame's column names, or `f0`, `f1`, ...
    min_coverage : float, default 0.95
        Minimum share of finite values, in `(0, 1]`.
    max_corr : float, default 0.95
        Absolute correlation above which one feature of a pair is dropped, in `(0, 1)`.

    Returns
    -------
    dict[str, Any]
        `table` (polars.DataFrame with columns `feature`, `status` (`"accepted"` or
        `"rejected"`), `coverage`, `std`, `max_abs_corr` and `reasons` (comma-separated),
        accepted first, then by coverage and std descending), `records` (`table` as a list
        of dicts), `selected_features` and `rejected_features` (in feature order),
        `rejection_reasons` (feature to list of reasons) and `meta` (`n_rows`,
        `n_features`, `min_coverage`, `max_corr`).

    Raises
    ------
    ValueError
        If a threshold is out of range, `X` is empty or not rectangular, or
        `feature_names` or `y` has the wrong length.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0.0 < max_corr < 1.0:
        raise ValueError("max_corr must be in (0, 1)")

    rows, names = _screen_rows(X, feature_names)
    n_rows = len(rows)
    n_features = len(names)
    if y is not None:
        y_vals = [float(v) for v in y]
        if len(y_vals) != n_rows:
            raise ValueError(f"y/X length mismatch: {len(y_vals)} vs {n_rows}")

    per_feature_vals: list[list[float]] = []
    coverage: list[float] = []
    stds: list[float] = []
    reasons: dict[str, list[str]] = {name: [] for name in names}

    for j, name in enumerate(names):
        vals = []
        for i in range(n_rows):
            v = rows[i][j]
            if v is None:
                continue
            vv = float(v)
            if not isfinite(vv):
                continue
            vals.append(vv)
        per_feature_vals.append(vals)
        cov = len(vals) / max(n_rows, 1)
        coverage.append(cov)
        sd = _std(vals)
        stds.append(sd)
        if cov < min_coverage:
            reasons[name].append(f"low_coverage<{min_coverage:.2f}")
        if sd <= _EPS:
            reasons[name].append("constant_or_near_constant")

    # Pairwise correlation with pairwise-valid rows only.
    corr = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    high_corr_pairs: list[tuple[int, int, float]] = []
    for i in range(n_features):
        corr[i][i] = 1.0
        for j in range(i + 1, n_features):
            c = _pairwise_corr(rows, i, j)
            corr[i][j] = c
            corr[j][i] = c
            if abs(c) > max_corr:
                high_corr_pairs.append((i, j, c))

    # Greedy correlated-feature rejection: drop weaker signal proxy first.
    dropped: set[int] = set()
    for i, j, c in sorted(high_corr_pairs, key=lambda t: abs(t[2]), reverse=True):
        if i in dropped or j in dropped:
            continue
        score_i = (coverage[i], stds[i])
        score_j = (coverage[j], stds[j])
        drop = j if score_i >= score_j else i
        keep = i if drop == j else j
        dropped.add(drop)
        reasons[names[drop]].append(f"high_corr>{max_corr:.2f}_with:{names[keep]}")

    rows_out = []
    selected_features: list[str] = []
    rejected_features: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}
    for idx, name in enumerate(names):
        max_abs_corr = (
            max(abs(corr[idx][j]) for j in range(n_features) if j != idx) if n_features > 1 else 0.0
        )
        rs = reasons[name]
        status = "accepted" if not rs else "rejected"
        rows_out.append(
            {
                "feature": name,
                "status": status,
                "coverage": coverage[idx],
                "std": stds[idx],
                "max_abs_corr": max_abs_corr,
                "reasons": ",".join(rs),
            }
        )
        if status == "accepted":
            selected_features.append(name)
        else:
            rejected_features.append(name)
            rejection_reasons[name] = rs

    table = pl.DataFrame(rows_out).sort(
        ["status", "coverage", "std"], descending=[False, True, True]
    )
    return {
        "table": table,
        "records": table.to_dicts(),
        "selected_features": selected_features,
        "rejected_features": rejected_features,
        "rejection_reasons": rejection_reasons,
        "meta": {
            "n_rows": n_rows,
            "n_features": n_features,
            "min_coverage": min_coverage,
            "max_corr": max_corr,
        },
    }
