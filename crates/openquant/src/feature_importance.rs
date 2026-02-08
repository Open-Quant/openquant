use std::collections::BTreeMap;

use nalgebra::{DMatrix, SymmetricEigen};

use crate::cross_validation::{ml_cross_val_score, Scoring, SimpleClassifier};

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
) -> Result<BTreeMap<String, ImportanceStats>, String> {
    if per_tree_importances.is_empty() {
        return Err("per_tree_importances cannot be empty".to_string());
    }
    let n_features = feature_names.len();
    if n_features == 0 {
        return Err("feature_names cannot be empty".to_string());
    }
    if per_tree_importances.iter().any(|r| r.len() != n_features) {
        return Err("importance row length mismatch".to_string());
    }

    let mut means = vec![0.0; n_features];
    let mut stderrs = vec![0.0; n_features];
    for j in 0..n_features {
        let col: Vec<f64> = per_tree_importances
            .iter()
            .map(|r| if r[j] == 0.0 { f64::NAN } else { r[j] })
            .collect();
        let (m, s) = nan_mean_std(&col);
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

pub fn mean_decrease_accuracy<C: SimpleClassifier>(
    model: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    feature_names: &[String],
    splits: &[(Vec<usize>, Vec<usize>)],
    sample_weight: Option<&[f64]>,
    scoring: Scoring,
) -> Result<BTreeMap<String, ImportanceStats>, String> {
    validate_xy(x, y, feature_names)?;

    let n_features = feature_names.len();
    let mut per_feature = vec![Vec::new(); n_features];

    for (train_idx, test_idx) in splits {
        let x_train = rows(x, train_idx);
        let y_train = vals(y, train_idx);
        let sw_train = sample_weight.map(|sw| vals(sw, train_idx));
        model.fit(&x_train, &y_train, sw_train.as_deref());

        let x_test = rows(x, test_idx);
        let y_test = vals(y, test_idx);
        let sw_test = sample_weight.map(|sw| vals(sw, test_idx));

        let base = score_model(model, &x_test, &y_test, sw_test.as_deref(), scoring);

        for j in 0..n_features {
            let mut x_perm = x_test.clone();
            permute_col(&mut x_perm, j);
            let perm = score_model(model, &x_perm, &y_test, sw_test.as_deref(), scoring);
            let imp = match scoring {
                Scoring::NegLogLoss => {
                    if -perm == 0.0 {
                        0.0
                    } else {
                        (base - perm) / (-perm)
                    }
                }
                Scoring::Accuracy => {
                    if (1.0 - perm).abs() < 1e-12 {
                        0.0
                    } else {
                        (base - perm) / (1.0 - perm)
                    }
                }
            };
            per_feature[j].push(if imp.is_finite() { imp } else { 0.0 });
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
) -> Result<BTreeMap<String, ImportanceStats>, String> {
    validate_xy(x, y, feature_names)?;
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        let xj: Vec<Vec<f64>> = x.iter().map(|r| vec![r[j]]).collect();
        let scores = ml_cross_val_score(clf, &xj, y, sample_weight, splits, scoring);
        let (mean, std) = mean_std(&scores);
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
) -> Result<Vec<Vec<f64>>, String> {
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
) -> Result<PcaCorrelation, String> {
    if feature_rows.is_empty() {
        return Err("feature_rows cannot be empty".to_string());
    }
    let n_features = feature_rows[0].len();
    if feature_importance_mean.len() != n_features {
        return Err("feature_importance_mean length mismatch".to_string());
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
    let inv_rank: Vec<f64> = pca_rank.iter().map(|r| 1.0 / *r as f64).collect();
    let weighted = weighted_kendall_tau(feature_importance_mean, &inv_rank);

    Ok(PcaCorrelation { pearson, spearman, kendall, weighted_kendall_rank: weighted })
}

pub fn plot_feature_importance(
    importance: &BTreeMap<String, ImportanceStats>,
    oob_score: f64,
    oos_score: f64,
    output_path: Option<&str>,
) -> Result<(), String> {
    if let Some(path) = output_path {
        let mut s = format!("oob_score,{oob_score}\noos_score,{oos_score}\nfeature,mean,std\n");
        for (k, v) in importance {
            s.push_str(&format!("{k},{},{}\n", v.mean, v.std));
        }
        std::fs::write(path, s).map_err(|e| format!("failed to write output file: {e}"))?;
    }
    Ok(())
}

fn compute_pca(
    feature_rows: &[Vec<f64>],
    variance_thresh: f64,
) -> Result<(Vec<f64>, DMatrix<f64>, Vec<Vec<f64>>), String> {
    if feature_rows.iter().any(|r| r.len() != feature_rows[0].len()) {
        return Err("ragged feature rows".to_string());
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

fn validate_xy(x: &[Vec<f64>], y: &[f64], feature_names: &[String]) -> Result<(), String> {
    if x.is_empty() || y.is_empty() {
        return Err("x and y cannot be empty".to_string());
    }
    if x.len() != y.len() {
        return Err("x/y length mismatch".to_string());
    }
    if x[0].len() != feature_names.len() {
        return Err("feature_names length mismatch".to_string());
    }
    if x.iter().any(|r| r.len() != x[0].len()) {
        return Err("ragged x rows".to_string());
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
    }
}

fn permute_col(x: &mut [Vec<f64>], col: usize) {
    if x.len() <= 1 {
        return;
    }
    let last = x[x.len() - 1][col];
    for i in (1..x.len()).rev() {
        x[i][col] = x[i - 1][col];
    }
    x[0][col] = last;
}

fn pack_stats(feature_names: &[String], values: &[Vec<f64>]) -> BTreeMap<String, ImportanceStats> {
    let mut out = BTreeMap::new();
    for (j, name) in feature_names.iter().enumerate() {
        let (m, s) = mean_std(&values[j]);
        let mean = if m.is_finite() { m } else { 0.0 };
        let std = if s.is_finite() { s * (values[j].len() as f64).powf(-0.5) } else { 0.0 };
        out.insert(name.clone(), ImportanceStats { mean, std });
    }
    out
}

fn nan_mean_std(v: &[f64]) -> (f64, f64) {
    let vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    mean_std(&vals)
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
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

fn rank_desc(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0usize; values.len()];
    for (r, i) in idx.iter().enumerate() {
        rank[*i] = r + 1;
    }
    rank
}

fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    let rx = rank_desc(x).iter().map(|r| *r as f64).collect::<Vec<_>>();
    let ry = rank_desc(y).iter().map(|r| *r as f64).collect::<Vec<_>>();
    pearson_corr(&rx, &ry)
}

fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let mut c = 0.0;
    let mut d = 0.0;
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let sx = (x[i] - x[j]).signum();
            let sy = (y[i] - y[j]).signum();
            let p = sx * sy;
            if p > 0.0 {
                c += 1.0;
            } else if p < 0.0 {
                d += 1.0;
            }
        }
    }
    let denom = c + d;
    if denom == 0.0 {
        0.0
    } else {
        (c - d) / denom
    }
}

fn weighted_kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let mut c = 0.0;
    let mut d = 0.0;
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let w = 1.0 / (1.0 + i as f64 + j as f64);
            let sx = (x[i] - x[j]).signum();
            let sy = (y[i] - y[j]).signum();
            let p = sx * sy;
            if p > 0.0 {
                c += w;
            } else if p < 0.0 {
                d += w;
            }
        }
    }
    let denom = c + d;
    if denom == 0.0 {
        0.0
    } else {
        (c - d) / denom
    }
}
