from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, isfinite, log, sqrt
from typing import Any, Sequence

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
        raise ValueError(
            f"feature_names length mismatch: expected {n_features}, got {len(out)}"
        )
    return out


def _sample_weight(weights: Sequence[float] | None, n_rows: int) -> list[float] | None:
    if weights is None:
        return None
    out = [float(v) for v in weights]
    if len(out) != n_rows:
        raise ValueError(
            f"sample_weight/X length mismatch: {len(out)} vs {n_rows}"
        )
    return out


def _build_intervals(event_end_indices: Sequence[int] | None, n_rows: int) -> list[tuple[int, int]]:
    if event_end_indices is None:
        return [(i, i) for i in range(n_rows)]
    ends = [int(v) for v in event_end_indices]
    if len(ends) != n_rows:
        raise ValueError(
            f"event_end_indices/X length mismatch: {len(ends)} vs {n_rows}"
        )
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

        before = max(0, start - embargo_n)
        after = min(n, stop + embargo_n)
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


def _fit_linear_probability_model(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    sample_weight: Sequence[float] | None,
    ridge: float = 1e-3,
) -> _LinearModel:
    n = len(x)
    p = len(x[0])
    sw = [1.0] * n if sample_weight is None else [float(v) for v in sample_weight]

    dim = p + 1
    xtwx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xtwy = [0.0 for _ in range(dim)]

    for row, yy, w in zip(x, y, sw):
        design = [1.0] + list(row)
        for i in range(dim):
            xtwy[i] += w * design[i] * yy
            for j in range(dim):
                xtwx[i][j] += w * design[i] * design[j]

    for i in range(dim):
        xtwx[i][i] += ridge

    beta = _solve_linear_system(xtwx, xtwy)
    return _LinearModel(coeffs=beta[1:], intercept=beta[0])


def _predict_proba(model: _LinearModel, x: Sequence[Sequence[float]]) -> list[float]:
    return [_sigmoid(model.intercept + _dot(row, model.coeffs)) for row in x]


def _score(
    y_true: Sequence[float],
    prob: Sequence[float],
    scoring: str,
    sample_weight: Sequence[float] | None,
) -> float:
    weights = [1.0] * len(y_true) if sample_weight is None else [float(v) for v in sample_weight]
    den = sum(weights)
    if den <= 0:
        return 0.0

    if scoring == "neg_log_loss":
        loss = 0.0
        for y, p, w in zip(y_true, prob, weights):
            p_clip = min(max(p, 1e-15), 1.0 - 1e-15)
            loss += w * (-(y * log(p_clip) + (1.0 - y) * log(1.0 - p_clip)))
        return -(loss / den)

    pred = [1.0 if p >= 0.5 else 0.0 for p in prob]

    if scoring == "accuracy":
        correct = sum(w for y, p, w in zip(y_true, pred, weights) if abs(y - p) < 1e-12)
        return correct / den

    if scoring == "f1":
        tp = sum(w for y, p, w in zip(y_true, pred, weights) if y > 0.5 and p > 0.5)
        fp = sum(w for y, p, w in zip(y_true, pred, weights) if y <= 0.5 and p > 0.5)
        fn = sum(w for y, p, w in zip(y_true, pred, weights) if y > 0.5 and p <= 0.5)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    raise ValueError("scoring must be one of: neg_log_loss, accuracy, f1")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return sqrt(var)


def _importance_table(feature_names: Sequence[str], per_feature_values: Sequence[Sequence[float]]) -> pl.DataFrame:
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


def mdi_importance(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str] | None = None,
    sample_weight: Sequence[float] | None = None,
    n_estimators: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    x = _as_matrix(X)
    yv = _as_vector(y, len(x))
    names = _feature_names(len(x[0]), feature_names)
    weights = _sample_weight(sample_weight, len(x))

    if n_estimators < 2:
        raise ValueError("n_estimators must be >= 2")

    import random

    rng = random.Random(seed)
    per_feature: list[list[float]] = [[] for _ in names]

    for _ in range(n_estimators):
        idx = [rng.randrange(len(x)) for _ in range(len(x))]
        xb = [x[i] for i in idx]
        yb = [yv[i] for i in idx]
        wb = [weights[i] for i in idx] if weights is not None else None

        model = _fit_linear_probability_model(xb, yb, wb)
        raw = [abs(v) for v in model.coeffs]
        denom = sum(raw)
        norm = [v / denom if denom > 0 else 0.0 for v in raw]
        for j, v in enumerate(norm):
            per_feature[j].append(v)

    table = _importance_table(names, per_feature)
    payload = viz.prepare_feature_importance_payload(
        table["feature"].to_list(), table["mean"].to_list(), std=table["stderr"].to_list()
    )
    return {
        "method": "mdi",
        "table": table,
        "records": table.to_dicts(),
        "viz_payload": payload,
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
    shift: int,
) -> tuple[float, list[float]]:
    model = _fit_linear_probability_model(x_train, y_train, sample_weight_train)
    base = _score(y_test, _predict_proba(model, x_test), scoring, sample_weight_test)

    out: list[float] = []
    n = len(x_test)
    s = shift % max(n, 1)
    for cols in groups:
        perm = [row[:] for row in x_test]
        for c in cols:
            col = [row[c] for row in x_test]
            shifted = col[-s:] + col[:-s] if s > 0 else col[:]
            for i in range(n):
                perm[i][c] = shifted[i]

        perm_score = _score(y_test, _predict_proba(model, perm), scoring, sample_weight_test)
        if scoring == "neg_log_loss":
            imp = (base - perm_score) / (-perm_score) if abs(perm_score) > _EPS else 0.0
        else:
            imp = (base - perm_score) / (1.0 - perm_score) if abs(1.0 - perm_score) > _EPS else 0.0
        out.append(imp if imp == imp and imp != float("inf") and imp != float("-inf") else 0.0)

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
) -> dict[str, object]:
    x = _as_matrix(X)
    yv = _as_vector(y, len(x))
    names = _feature_names(len(x[0]), feature_names)
    weights = _sample_weight(sample_weight, len(x))
    intervals = _build_intervals(event_end_indices, len(x))
    splits = _purged_kfold_splits(intervals, n_splits=n_splits, pct_embargo=pct_embargo)

    per_feature: list[list[float]] = [[] for _ in names]
    fold_scores: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        x_train = [x[i] for i in train_idx]
        y_train = [yv[i] for i in train_idx]
        x_test = [x[i] for i in test_idx]
        y_test = [yv[i] for i in test_idx]
        w_train = [weights[i] for i in train_idx] if weights is not None else None
        w_test = [weights[i] for i in test_idx] if weights is not None else None

        base, scores = _score_with_perm_groups(
            x_train,
            y_train,
            x_test,
            y_test,
            [[j] for j in range(len(names))],
            scoring,
            w_train,
            w_test,
            shift=fold_idx + 1,
        )
        fold_scores.append(base)
        for j, imp in enumerate(scores):
            per_feature[j].append(imp)

    table = _importance_table(names, per_feature)
    payload = viz.prepare_feature_importance_payload(
        table["feature"].to_list(), table["mean"].to_list(), std=table["stderr"].to_list()
    )
    return {
        "method": "mda",
        "table": table,
        "records": table.to_dicts(),
        "viz_payload": payload,
        "cv": {
            "method": "purged_kfold",
            "n_splits": n_splits,
            "pct_embargo": pct_embargo,
            "fold_count": len(splits),
            "scoring": scoring,
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
) -> dict[str, object]:
    x = _as_matrix(X)
    yv = _as_vector(y, len(x))
    names = _feature_names(len(x[0]), feature_names)
    weights = _sample_weight(sample_weight, len(x))
    intervals = _build_intervals(event_end_indices, len(x))
    splits = _purged_kfold_splits(intervals, n_splits=n_splits, pct_embargo=pct_embargo)

    per_feature: list[list[float]] = [[] for _ in names]

    for j in range(len(names)):
        xj = [[row[j]] for row in x]
        for train_idx, test_idx in splits:
            x_train = [xj[i] for i in train_idx]
            y_train = [yv[i] for i in train_idx]
            x_test = [xj[i] for i in test_idx]
            y_test = [yv[i] for i in test_idx]
            w_train = [weights[i] for i in train_idx] if weights is not None else None
            w_test = [weights[i] for i in test_idx] if weights is not None else None

            model = _fit_linear_probability_model(x_train, y_train, w_train)
            score_val = _score(y_test, _predict_proba(model, x_test), scoring, w_test)
            per_feature[j].append(score_val)

    table = _importance_table(names, per_feature)
    payload = viz.prepare_feature_importance_payload(
        table["feature"].to_list(), table["mean"].to_list(), std=table["stderr"].to_list()
    )
    return {
        "method": "sfi",
        "table": table,
        "records": table.to_dicts(),
        "viz_payload": payload,
        "cv": {
            "method": "purged_kfold",
            "n_splits": n_splits,
            "pct_embargo": pct_embargo,
            "fold_count": len(splits),
            "scoring": scoring,
        },
    }


def _standardize(x: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    n = len(x)
    p = len(x[0])
    means = [sum(row[j] for row in x) / n for j in range(p)]
    stds = []
    for j in range(p):
        var = sum((row[j] - means[j]) ** 2 for row in x) / max(n - 1, 1)
        stds.append(sqrt(var) if var > 0 else 1.0)

    z = [[(row[j] - means[j]) / stds[j] for j in range(p)] for row in x]
    return z, means, stds


def _mat_vec(a: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [sum(aij * vj for aij, vj in zip(row, v)) for row in a]


def _norm(v: Sequence[float]) -> float:
    return sqrt(sum(x * x for x in v))


def _power_iteration(a: list[list[float]], iters: int = 200) -> tuple[float, list[float]]:
    n = len(a)
    v = [1.0 / sqrt(n)] * n
    for _ in range(iters):
        av = _mat_vec(a, v)
        nrm = _norm(av)
        if nrm <= _EPS:
            break
        v = [x / nrm for x in av]
    av = _mat_vec(a, v)
    eig = _dot(v, av)
    return eig, v


def orthogonalize_features_pca(
    X: Sequence[Sequence[float]],
    variance_threshold: float = 0.95,
) -> dict[str, object]:
    if variance_threshold <= 0.0 or variance_threshold > 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")

    x = _as_matrix(X)
    z, means, stds = _standardize(x)
    n = len(z)
    p = len(z[0])

    cov = [[0.0 for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(p):
            cov[i][j] = sum(row[i] * row[j] for row in z) / max(n - 1, 1)

    work = [row[:] for row in cov]
    eigvals: list[float] = []
    eigvecs: list[list[float]] = []

    for _ in range(p):
        eig, vec = _power_iteration(work)
        if eig <= _EPS:
            break
        eigvals.append(eig)
        eigvecs.append(vec)
        # deflation: A = A - lambda * v v^T
        for i in range(p):
            for j in range(p):
                work[i][j] -= eig * vec[i] * vec[j]

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

    columns = {f"pc{i+1}": [row[i] for row in transformed] for i in range(kept)}
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
    n = len(z)
    p = len(z[0])
    corr = [[0.0 for _ in range(p)] for _ in range(p)]
    for i in range(p):
        for j in range(p):
            corr[i][j] = sum(row[i] * row[j] for row in z) / max(n - 1, 1)
    return corr


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
) -> dict[str, object]:
    x = _as_matrix(X)
    yv = _as_vector(y, len(x))
    names = _feature_names(len(x[0]), feature_names)
    weights = _sample_weight(sample_weight, len(x))
    intervals = _build_intervals(event_end_indices, len(x))
    splits = _purged_kfold_splits(intervals, n_splits=n_splits, pct_embargo=pct_embargo)

    mda = mda_importance(
        x,
        yv,
        feature_names=names,
        sample_weight=weights,
        event_end_indices=event_end_indices,
        n_splits=n_splits,
        pct_embargo=pct_embargo,
        scoring=scoring,
    )
    base_table: pl.DataFrame = mda["table"]
    base_map = {row["feature"]: float(row["mean"]) for row in base_table.to_dicts()}

    corr = _corr_matrix(x)
    pairs: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr_ij = corr[i][j]
            if abs(corr_ij) < corr_threshold:
                continue

            grouped_vals: list[float] = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                x_train = [x[k] for k in train_idx]
                y_train = [yv[k] for k in train_idx]
                x_test = [x[k] for k in test_idx]
                y_test = [yv[k] for k in test_idx]
                w_train = [weights[k] for k in train_idx] if weights is not None else None
                w_test = [weights[k] for k in test_idx] if weights is not None else None
                _, group_imp = _score_with_perm_groups(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    [[i, j]],
                    scoring,
                    w_train,
                    w_test,
                    shift=fold_idx + 1,
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

    pairs_df = pl.DataFrame(pairs) if pairs else pl.DataFrame(
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

    out: dict[str, object] = {
        "baseline_mda": mda,
        "pairs": pairs_df,
        "pair_records": pairs_df.to_dicts(),
    }

    if orthogonalize:
        ortho = orthogonalize_features_pca(x, variance_threshold=0.95)
        x_ortho = ortho["transformed"]
        pc_names = [f"pc{i+1}" for i in range(len(x_ortho[0]))]
        ortho_mda = mda_importance(
            x_ortho,
            yv,
            feature_names=pc_names,
            sample_weight=weights,
            event_end_indices=event_end_indices,
            n_splits=n_splits,
            pct_embargo=pct_embargo,
            scoring=scoring,
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


def feature_screen_report(
    X: Sequence[Sequence[float]] | pl.DataFrame,
    y: Sequence[float] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    min_coverage: float = 0.95,
    max_corr: float = 0.95,
) -> dict[str, object]:
    """Run lightweight feature QA checks for notebook discovery loops."""
    if min_coverage <= 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if max_corr <= 0.0 or max_corr >= 1.0:
        raise ValueError("max_corr must be in (0, 1)")

    if isinstance(X, pl.DataFrame):
        if X.width == 0 or X.height == 0:
            raise ValueError("X cannot be empty")
        names = list(feature_names) if feature_names is not None else [str(c) for c in X.columns]
        if feature_names is not None and len(names) != X.width:
            raise ValueError(f"feature_names length mismatch: expected {X.width}, got {len(names)}")
        rows = X.select([pl.col(c) for c in X.columns]).rows()
    else:
        rows = [list(r) for r in X]
        if not rows:
            raise ValueError("X cannot be empty")
        width = len(rows[0])
        if width == 0:
            raise ValueError("X must contain at least one feature")
        if any(len(r) != width for r in rows):
            raise ValueError("X must be rectangular")
        names = _feature_names(width, feature_names)

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
            pairs: list[tuple[float, float]] = []
            for k in range(n_rows):
                vi = rows[k][i]
                vj = rows[k][j]
                if vi is None or vj is None:
                    continue
                fi = float(vi)
                fj = float(vj)
                if isfinite(fi) and isfinite(fj):
                    pairs.append((fi, fj))

            if len(pairs) < 2:
                c = 0.0
            else:
                ai = [p[0] for p in pairs]
                aj = [p[1] for p in pairs]
                mi = _mean(ai)
                mj = _mean(aj)
                sdi = _std(ai)
                sdj = _std(aj)
                if sdi <= _EPS or sdj <= _EPS:
                    c = 0.0
                else:
                    cov = sum((u - mi) * (v - mj) for u, v in pairs) / (len(pairs) - 1)
                    c = cov / (sdi * sdj)
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
        max_abs_corr = max(abs(corr[idx][j]) for j in range(n_features) if j != idx) if n_features > 1 else 0.0
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

    table = pl.DataFrame(rows_out).sort(["status", "coverage", "std"], descending=[False, True, True])
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
