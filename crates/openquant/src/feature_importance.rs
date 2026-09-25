use std::collections::BTreeMap;

use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::cross_validation::{ml_cross_val_score, Scoring, SimpleClassifier};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FeatureImportanceError {
    #[error("failed to write output file: {0}")]
    WriteOutput(String),
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    #[error("importance row length mismatch")]
    ImportanceRowLengthMismatch,
    #[error("{0} length mismatch")]
    LengthMismatch(&'static str),
    #[error("ragged feature rows")]
    RaggedFeatureRows,
    #[error("x and y cannot be empty")]
    EmptyXy,
    #[error("x/y length mismatch")]
    XyLengthMismatch,
    #[error("ragged x rows")]
    RaggedX,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImportanceStats {
    pub mean: f64,
    pub std: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PcaCorrelation {
    pub pearson: f64,
    pub spearman: f64,
    pub kendall: f64,
    pub weighted_kendall_rank: f64,
}

pub fn mean_decrease_impurity(
    per_tree_importances: &[Vec<f64>],
    feature_names: &[String],
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    if per_tree_importances.is_empty() {
        return Err(FeatureImportanceError::Empty("per_tree_importances"));
    }
    let n_features = feature_names.len();
    if n_features == 0 {
        return Err(FeatureImportanceError::Empty("feature_names"));
    }
    if per_tree_importances.iter().any(|r| r.len() != n_features) {
        return Err(FeatureImportanceError::ImportanceRowLengthMismatch);
    }

    let mut means = vec![0.0; n_features];
    let mut stderrs = vec![0.0; n_features];
    for j in 0..n_features {
        let col: Vec<f64> = per_tree_importances
            .iter()
            .map(|r| if r[j] == 0.0 { f64::NAN } else { r[j] })
            .collect();
        // Snippet 8.2: pandas `df0.std()`, the sample deviation (ddof = 1).
        let (m, s) = nan_mean_std(&col, 1);
        means[j] = m;
        stderrs[j] = s * (per_tree_importances.len() as f64).powf(-0.5);
    }

    let denom: f64 = means.iter().filter(|v| v.is_finite()).sum();
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        let mean = if denom > 0.0 && means[j].is_finite() { means[j] / denom } else { 0.0 };
        let std = if denom > 0.0 && stderrs[j].is_finite() { stderrs[j] / denom } else { 0.0 };
        out.insert(name.clone(), ImportanceStats { mean, std });
    }
    Ok(out)
}

/// Mean decrease accuracy (AFML Snippet 8.3): for each split, fit on the train rows, score the
/// test rows, then score them again with one feature column shuffled; importance is the relative
/// loss of score. Shuffles draw from a `StdRng` seeded with `seed`, so a given seed always
/// gives the same result.
#[allow(clippy::too_many_arguments)]
pub fn mean_decrease_accuracy<C: SimpleClassifier>(
    model: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
    splits: &[(Vec<usize>, Vec<usize>)],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
    seed: u64,
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    validate_xy(x, y, feature_names)?;

    let n_features = feature_names.len();
    let mut per_feature = vec![Vec::new(); n_features];
    let mut rng = StdRng::seed_from_u64(seed);

    for (train_idx, test_idx) in splits {
        let x_train = rows(x, train_idx);
        let y_train = vals(y, train_idx);
        let sw_train = sample_weight.map(|sw| vals(sw, train_idx));
        model.fit(&x_train, &y_train, sw_train.as_deref());

        let x_test = rows(x, test_idx);
        let y_test = vals(y, test_idx);
        let sw_test = sample_weight.map(|sw| vals(sw, test_idx));

        let base = score_model(model, &x_test, &y_test, sw_test.as_deref(), scoring);

        for (j, scores) in per_feature.iter_mut().enumerate() {
            let mut x_perm = x_test.clone();
            permute_col(&mut x_perm, j, &mut rng);
            let perm = score_model(model, &x_perm, &y_test, sw_test.as_deref(), scoring);
            let imp = match scoring {
                Scoring::NegLogLoss => {
                    if -perm == 0.0 {
                        0.0
                    } else {
                        (base - perm) / (-perm)
                    }
                }
                Scoring::Accuracy | Scoring::F1 => {
                    if (1.0 - perm).abs() < 1e-12 {
                        0.0
                    } else {
                        (base - perm) / (1.0 - perm)
                    }
                }
            };
            scores.push(if imp.is_finite() { imp } else { 0.0 });
        }
    }

    Ok(pack_stats(feature_names, &per_feature))
}

pub fn single_feature_importance<C: SimpleClassifier>(
    clf: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
    splits: &[(Vec<usize>, Vec<usize>)],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
) -> Result<BTreeMap<String, ImportanceStats>, FeatureImportanceError> {
    validate_xy(x, y, feature_names)?;
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        let xj: Vec<Vec<f64>> = x.iter().map(|r| vec![r[j]]).collect();
        let scores = ml_cross_val_score(clf, &xj, y, sample_weight, splits, scoring);
        // Snippet 8.4 takes `.std()` of the numpy array cvScore returns: ddof = 0.
        let (mean, std) = mean_std(&scores, 0);
        out.insert(
            name.clone(),
            ImportanceStats { mean, std: std * (scores.len() as f64).powf(-0.5) },
        );
    }
    Ok(out)
}

pub fn get_orthogonal_features(
    feature_rows: &[Vec<f64>],
    variance_thresh: f64,
) -> Result<Vec<Vec<f64>>, FeatureImportanceError> {
    if feature_rows.is_empty() {
        return Ok(Vec::new());
    }
    let (_, evec, x_std) = compute_pca(feature_rows, variance_thresh)?;
    Ok((to_dmatrix(&x_std) * evec).row_iter().map(|r| r.iter().copied().collect()).collect())
}

pub fn feature_pca_analysis(
    feature_rows: &[Vec<f64>],
    feature_importance_mean: &[f64],
    variance_thresh: f64,
) -> Result<PcaCorrelation, FeatureImportanceError> {
    if feature_rows.is_empty() {
        return Err(FeatureImportanceError::Empty("feature_rows"));
    }
    let n_features = feature_rows[0].len();
    if feature_importance_mean.len() != n_features {
        return Err(FeatureImportanceError::LengthMismatch("feature_importance_mean"));
    }

    let (eval, evec, _) = compute_pca(feature_rows, variance_thresh)?;

    let pcs = eval.len();
    let mut all_eigs = Vec::with_capacity(n_features * pcs);
    for c in 0..pcs {
        for r in 0..n_features {
            all_eigs.push((evec[(r, c)] * eval[c]).abs());
        }
    }
    let mut repeated_imp = Vec::with_capacity(n_features * pcs);
    for _ in 0..pcs {
        repeated_imp.extend_from_slice(feature_importance_mean);
    }

    let pearson = pearson_corr(&repeated_imp, &all_eigs);
    let spearman = spearman_corr(&repeated_imp, &all_eigs);
    let kendall = kendall_tau(&repeated_imp, &all_eigs);

    let mut pca_strength = vec![0.0; n_features];
    for r in 0..n_features {
        let mut s = 0.0;
        for c in 0..pcs {
            s += (evec[(r, c)] * eval[c]).abs();
        }
        pca_strength[r] = s;
    }
    let pca_rank = rank_desc(&pca_strength);
    let inv_rank: Vec<f64> = pca_rank.iter().map(|r| 1.0 / r).collect();
    let weighted = weighted_kendall_tau(feature_importance_mean, &inv_rank);

    Ok(PcaCorrelation { pearson, spearman, kendall, weighted_kendall_rank: weighted })
}

pub fn plot_feature_importance(
    importance: &BTreeMap<String, ImportanceStats>,
    oob_score: f64,
    oos_score: f64,
    output_path: Option<&str>,
) -> Result<(), FeatureImportanceError> {
    if let Some(path) = output_path {
        let mut s = format!("oob_score,{oob_score}\noos_score,{oos_score}\nfeature,mean,std\n");
        for (k, v) in importance {
            s.push_str(&format!("{k},{},{}\n", v.mean, v.std));
        }
        std::fs::write(path, s).map_err(|e| FeatureImportanceError::WriteOutput(e.to_string()))?;
    }
    Ok(())
}

/// PCA output: `(eigenvalues, eigenvectors, standardized feature rows)`.
type PcaDecomposition = (Vec<f64>, DMatrix<f64>, Vec<Vec<f64>>);

fn compute_pca(
    feature_rows: &[Vec<f64>],
    variance_thresh: f64,
) -> Result<PcaDecomposition, FeatureImportanceError> {
    if feature_rows.iter().any(|r| r.len() != feature_rows[0].len()) {
        return Err(FeatureImportanceError::RaggedFeatureRows);
    }
    let x_std = standardize(feature_rows);
    let x = to_dmatrix(&x_std);
    let dot = x.transpose() * &x;
    let eig = SymmetricEigen::new(dot);

    let mut idx: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    idx.sort_by(|&a, &b| {
        eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut eval = Vec::with_capacity(idx.len());
    let mut evec_cols = Vec::with_capacity(idx.len());
    for i in idx {
        eval.push(eig.eigenvalues[i]);
        evec_cols.push(eig.eigenvectors.column(i).clone_owned());
    }

    let total: f64 = eval.iter().sum();
    let mut cum = 0.0;
    let mut dim = 0usize;
    if total > 0.0 {
        for (i, v) in eval.iter().enumerate() {
            cum += *v;
            dim = i;
            if cum / total >= variance_thresh {
                break;
            }
        }
    }
    let kept = dim + 1;
    eval.truncate(kept);
    let evec = DMatrix::<f64>::from_columns(&evec_cols[..kept]);
    Ok((eval, evec, x_std))
}

fn validate_xy(
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
) -> Result<(), FeatureImportanceError> {
    if x.is_empty() || y.is_empty() {
        return Err(FeatureImportanceError::EmptyXy);
    }
    if x.len() != y.len() {
        return Err(FeatureImportanceError::XyLengthMismatch);
    }
    if x[0].len() != feature_names.len() {
        return Err(FeatureImportanceError::LengthMismatch("feature_names"));
    }
    if x.iter().any(|r| r.len() != x[0].len()) {
        return Err(FeatureImportanceError::RaggedX);
    }
    Ok(())
}

fn rows(x: &[Vec<f64>], idx: &[usize]) -> Vec<Vec<f64>> {
    idx.iter().map(|i| x[*i].clone()).collect()
}

fn vals(v: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|i| v[*i]).collect()
}

fn score_model<C: SimpleClassifier>(
    model: &C,
    x_test: &[Vec<f64>],
    y_test: &[f64],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
) -> f64 {
    match scoring {
        Scoring::Accuracy => {
            let pred = model.predict(x_test);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..y_test.len() {
                let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
                den += w;
                if (pred[i] - y_test[i]).abs() < 1e-12 {
                    num += w;
                }
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }
        Scoring::NegLogLoss => {
            let probs = model.predict_proba(x_test);
            let mut loss = 0.0;
            let mut den = 0.0;
            let eps = 1e-15;
            for i in 0..y_test.len() {
                let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
                let p = probs[i].clamp(eps, 1.0 - eps);
                loss += w * (-(y_test[i] * p.ln() + (1.0 - y_test[i]) * (1.0 - p).ln()));
                den += w;
            }
            if den > 0.0 {
                -(loss / den)
            } else {
                0.0
            }
        }
        Scoring::F1 => {
            let pred = model.predict(x_test);
            let mut tp = 0.0;
            let mut fp = 0.0;
            let mut fnn = 0.0;
            for i in 0..y_test.len() {
                let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
                let p_pos = pred[i] > 0.5;
                let y_pos = y_test[i] > 0.5;
                if p_pos && y_pos {
                    tp += w;
                } else if p_pos && !y_pos {
                    fp += w;
                } else if !p_pos && y_pos {
                    fnn += w;
                }
            }
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fnn > 0.0 { tp / (tp + fnn) } else { 0.0 };
            if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            }
        }
    }
}

/// Shuffles one column of `x` in place, as AFML Snippet 8.3 does with `np.random.shuffle`.
/// A shuffle (unlike a rotation) breaks the feature-label link however persistent the feature is.
fn permute_col(x: &mut [Vec<f64>], col: usize, rng: &mut StdRng) {
    let mut values: Vec<f64> = x.iter().map(|row| row[col]).collect();
    values.shuffle(rng);
    for (row, v) in x.iter_mut().zip(values) {
        row[col] = v;
    }
}

fn pack_stats(feature_names: &[String], values: &[Vec<f64>]) -> BTreeMap<String, ImportanceStats> {
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        // Snippet 8.3: `imp.std()` on a pandas DataFrame, the sample deviation (ddof = 1).
        let (m, s) = mean_std(&values[j], 1);
        let mean = if m.is_finite() { m } else { 0.0 };
        let std = if s.is_finite() { s * (values[j].len() as f64).powf(-0.5) } else { 0.0 };
        out.insert(name.clone(), ImportanceStats { mean, std });
    }
    out
}

fn nan_mean_std(v: &[f64], ddof: usize) -> (f64, f64) {
    let vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    mean_std(&vals, ddof)
}

/// Mean and standard deviation with `ddof` delta degrees of freedom (divide by `n - ddof`).
/// The deviation is 0 when there are no more than `ddof` values (pandas would give NaN).
fn mean_std(v: &[f64], ddof: usize) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    if v.len() <= ddof {
        return (mean, 0.0);
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - ddof) as f64;
    (mean, var.sqrt())
}

fn standardize(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len();
    let m = rows[0].len();
    let mut means = vec![0.0; m];
    for r in rows {
        for j in 0..m {
            means[j] += r[j];
        }
    }
    for v in &mut means {
        *v /= n as f64;
    }
    let mut stds = vec![0.0; m];
    for r in rows {
        for j in 0..m {
            stds[j] += (r[j] - means[j]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / n as f64).sqrt();
    }

    rows.iter()
        .map(|r| {
            (0..m).map(|j| if stds[j] > 0.0 { (r[j] - means[j]) / stds[j] } else { 0.0 }).collect()
        })
        .collect()
}

fn to_dmatrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    DMatrix::<f64>::from_row_slice(n, m, &flat)
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let mut num = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        num / (vx.sqrt() * vy.sqrt())
    }
}

/// Ranks with 1 for the largest value; tied values share the average of their ranks, as
/// pandas `rank(ascending=False)` and `scipy.stats.rankdata(-v)` do.
fn rank_desc(values: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0.0; values.len()];
    let mut first = 0;
    while first < idx.len() {
        let mut last = first;
        while last + 1 < idx.len() && values[idx[last + 1]] == values[idx[first]] {
            last += 1;
        }
        // positions first..=last hold ranks first+1 ..= last+1
        let avg = (first + last) as f64 / 2.0 + 1.0;
        for i in &idx[first..=last] {
            rank[*i] = avg;
        }
        first = last + 1;
    }
    rank
}

/// `scipy.stats.spearmanr`: Pearson correlation of average ranks.
fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    pearson_corr(&rank_desc(x), &rank_desc(y))
}

/// `scipy.stats.kendalltau` (the default tau-b): `(C - D) / sqrt((P - T_x) (P - T_y))`, where
/// `P` counts all pairs and `T_x`, `T_y` the pairs tied in x and in y. 0 when either input is
/// constant (scipy returns NaN).
fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let (mut s, mut untied_x, mut untied_y) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let sx = sign(x[i] - x[j]);
            let sy = sign(y[i] - y[j]);
            s += sx * sy;
            untied_x += sx.abs();
            untied_y += sy.abs();
        }
    }
    if untied_x == 0.0 || untied_y == 0.0 {
        0.0
    } else {
        s / (untied_x.sqrt() * untied_y.sqrt())
    }
}

/// `scipy.stats.weightedtau` with its defaults (AFML Snippet 8.6): Vigna's weighted tau with
/// additive hyperbolic weights, averaged over ranking the elements by `(x, y)` and by `(y, x)`.
/// 0 when either input is constant (scipy returns NaN).
fn weighted_kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    match (weighted_tau_ranked(x, y), weighted_tau_ranked(y, x)) {
        (Some(a), Some(b)) => (a + b) / 2.0,
        _ => 0.0,
    }
}

/// One half of `weightedtau`: the element with the largest `x` (ties broken by larger `y`, then
/// by larger index, as scipy's reversed `lexsort` does) has rank 0 and weight 1, the next
/// weight 1/2, and so on. A pair weighs the sum of its two elements' weights. The result is
/// `sum_pairs w * sgn(dx) * sgn(dy) / sqrt(sum_{dx != 0} w * sum_{dy != 0} w)`.
fn weighted_tau_ranked(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.cmp(&b))
    });
    let mut weight = vec![0.0; n];
    for (rank, i) in order.iter().rev().enumerate() {
        weight[*i] = 1.0 / (rank as f64 + 1.0);
    }

    let (mut s, mut untied_x, mut untied_y) = (0.0, 0.0, 0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let w = weight[i] + weight[j];
            let sx = sign(x[i] - x[j]);
            let sy = sign(y[i] - y[j]);
            s += w * sx * sy;
            untied_x += w * sx.abs();
            untied_y += w * sy.abs();
        }
    }
    if untied_x == 0.0 || untied_y == 0.0 {
        return None;
    }
    Some((s / (untied_x.sqrt() * untied_y.sqrt())).clamp(-1.0, 1.0))
}

/// -1, 0 or 1. Unlike `f64::signum`, 0.0 maps to 0 so that ties count as ties.
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both inputs have ties: x at 0.1 and 0.3, y at 2.0 and 3.0. Expected values from scipy 1.18:
    //   spearmanr(x, y) = 0.5454545454545454
    //   kendalltau(x, y) = 0.3846153846153847   (tau-b: S = 5 over 15 pairs, 2 tied in each)
    //   weightedtau(x, y) = 0.23914127716864048
    const X: [f64; 6] = [0.3, 0.1, 0.3, 0.2, 0.1, 0.4];
    const Y: [f64; 6] = [2.0, 1.0, 3.0, 3.0, 0.5, 2.0];

    #[test]
    fn average_ranks_for_ties() {
        assert_eq!(rank_desc(&X), vec![2.5, 5.5, 2.5, 4.0, 5.5, 1.0]);
    }

    #[test]
    fn rank_correlations_match_scipy_with_ties() {
        assert!((spearman_corr(&X, &Y) - 0.5454545454545454).abs() < 1e-12);
        assert!((kendall_tau(&X, &Y) - 5.0 / 13.0).abs() < 1e-12);
        assert!((weighted_kendall_tau(&X, &Y) - 0.23914127716864048).abs() < 1e-12);
    }

    #[test]
    fn rank_correlations_of_a_constant_are_zero() {
        let c = [1.0; 6];
        assert_eq!(kendall_tau(&c, &Y), 0.0);
        assert_eq!(weighted_kendall_tau(&c, &Y), 0.0);
    }
}
