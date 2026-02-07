use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct Effect {
    pub raw: BTreeMap<usize, f64>,
    pub norm: BTreeMap<usize, f64>,
}

#[derive(Clone, Debug, Default)]
pub struct PairwiseEffect {
    pub raw: BTreeMap<String, f64>,
    pub norm: BTreeMap<String, f64>,
}

pub trait RegressionPredictor {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64>;
}

pub trait ClassificationPredictor {
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64>;
}

#[derive(Clone, Debug, Default)]
pub struct RegressionModelFingerprint {
    linear_effect: Option<Effect>,
    non_linear_effect: Option<Effect>,
    pair_wise_effect: Option<PairwiseEffect>,
}

#[derive(Clone, Debug, Default)]
pub struct ClassificationModelFingerprint {
    linear_effect: Option<Effect>,
    non_linear_effect: Option<Effect>,
    pair_wise_effect: Option<PairwiseEffect>,
}

impl RegressionModelFingerprint {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit<M: RegressionPredictor>(
        &mut self,
        model: &M,
        x: &[Vec<f64>],
        num_values: usize,
        pairwise_combinations: Option<&[(usize, usize)]>,
    ) -> Result<(), String> {
        let (lin, nonlin, pair) = fit_impl(
            |data| model.predict(data),
            x,
            num_values,
            pairwise_combinations,
        )?;
        self.linear_effect = Some(lin);
        self.non_linear_effect = Some(nonlin);
        self.pair_wise_effect = pair;
        Ok(())
    }

    pub fn get_effects(&self) -> Result<(&Effect, &Effect, Option<&PairwiseEffect>), String> {
        let lin = self
            .linear_effect
            .as_ref()
            .ok_or_else(|| "fit must be called before get_effects".to_string())?;
        let nonlin = self
            .non_linear_effect
            .as_ref()
            .ok_or_else(|| "fit must be called before get_effects".to_string())?;
        Ok((lin, nonlin, self.pair_wise_effect.as_ref()))
    }

    pub fn plot_effects(&self) -> Result<Vec<String>, String> {
        let (lin, nonlin, pair) = self.get_effects()?;
        let mut lines = vec![
            format!("linear:{} features", lin.raw.len()),
            format!("nonlinear:{} features", nonlin.raw.len()),
        ];
        if let Some(p) = pair {
            lines.push(format!("pairwise:{} pairs", p.raw.len()));
        }
        Ok(lines)
    }
}

impl ClassificationModelFingerprint {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit<M: ClassificationPredictor>(
        &mut self,
        model: &M,
        x: &[Vec<f64>],
        num_values: usize,
        pairwise_combinations: Option<&[(usize, usize)]>,
    ) -> Result<(), String> {
        let (lin, nonlin, pair) = fit_impl(
            |data| model.predict_proba(data),
            x,
            num_values,
            pairwise_combinations,
        )?;
        self.linear_effect = Some(lin);
        self.non_linear_effect = Some(nonlin);
        self.pair_wise_effect = pair;
        Ok(())
    }

    pub fn get_effects(&self) -> Result<(&Effect, &Effect, Option<&PairwiseEffect>), String> {
        let lin = self
            .linear_effect
            .as_ref()
            .ok_or_else(|| "fit must be called before get_effects".to_string())?;
        let nonlin = self
            .non_linear_effect
            .as_ref()
            .ok_or_else(|| "fit must be called before get_effects".to_string())?;
        Ok((lin, nonlin, self.pair_wise_effect.as_ref()))
    }

    pub fn plot_effects(&self) -> Result<Vec<String>, String> {
        let (lin, nonlin, pair) = self.get_effects()?;
        let mut lines = vec![
            format!("linear:{} features", lin.raw.len()),
            format!("nonlinear:{} features", nonlin.raw.len()),
        ];
        if let Some(p) = pair {
            lines.push(format!("pairwise:{} pairs", p.raw.len()));
        }
        Ok(lines)
    }
}

fn fit_impl<F>(
    predictor: F,
    x: &[Vec<f64>],
    num_values: usize,
    pairwise_combinations: Option<&[(usize, usize)]>,
) -> Result<(Effect, Effect, Option<PairwiseEffect>), String>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    if x.is_empty() {
        return Err("x cannot be empty".to_string());
    }
    if num_values < 2 {
        return Err("num_values must be >= 2".to_string());
    }
    let n_features = x[0].len();
    if n_features == 0 {
        return Err("x must have at least one feature".to_string());
    }
    if x.iter().any(|r| r.len() != n_features) {
        return Err("ragged x rows".to_string());
    }

    let feature_values = get_feature_values(x, num_values);
    let partial_dep = get_individual_partial_dependence(&predictor, x, &feature_values);
    let linear_raw = get_linear_effect(&feature_values, &partial_dep);
    let nonlin_raw = get_non_linear_effect(&feature_values, &partial_dep);
    let linear = Effect {
        norm: normalize_usize_map(&linear_raw),
        raw: linear_raw,
    };
    let non_linear = Effect {
        norm: normalize_usize_map(&nonlin_raw),
        raw: nonlin_raw,
    };

    let pair = pairwise_combinations.map(|pairs| {
        let raw = get_pairwise_effect(pairs, &predictor, x, num_values, &feature_values, &partial_dep);
        PairwiseEffect {
            norm: normalize_string_map(&raw),
            raw,
        }
    });

    Ok((linear, non_linear, pair))
}

fn get_feature_values(x: &[Vec<f64>], num_values: usize) -> Vec<Vec<f64>> {
    let n_features = x[0].len();
    let mut out = vec![vec![0.0; num_values]; n_features];
    for j in 0..n_features {
        let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (k, q) in (0..num_values)
            .map(|i| i as f64 / (num_values - 1) as f64)
            .enumerate()
        {
            out[j][k] = quantile_sorted(&col, q);
        }
    }
    out
}

fn get_individual_partial_dependence<F>(
    predictor: &F,
    x: &[Vec<f64>],
    feature_values: &[Vec<f64>],
) -> Vec<Vec<f64>>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    let n_features = x[0].len();
    let num_values = feature_values[0].len();
    let mut out = vec![vec![0.0; num_values]; n_features];
    for j in 0..n_features {
        for (k, &xk) in feature_values[j].iter().enumerate() {
            let mut x_mod = x.to_vec();
            for row in &mut x_mod {
                row[j] = xk;
            }
            let pred = predictor(&x_mod);
            out[j][k] = pred.iter().sum::<f64>() / pred.len() as f64;
        }
    }
    out
}

fn get_linear_effect(feature_values: &[Vec<f64>], partial_dep: &[Vec<f64>]) -> BTreeMap<usize, f64> {
    let mut store = BTreeMap::new();
    for j in 0..feature_values.len() {
        let x = &feature_values[j];
        let y = &partial_dep[j];
        let (a, b) = ols_line(x, y);
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let effect = x
            .iter()
            .map(|v| (a + b * *v - y_mean).abs())
            .sum::<f64>()
            / x.len() as f64;
        store.insert(j, effect);
    }
    store
}

fn get_non_linear_effect(feature_values: &[Vec<f64>], partial_dep: &[Vec<f64>]) -> BTreeMap<usize, f64> {
    let mut store = BTreeMap::new();
    for j in 0..feature_values.len() {
        let x = &feature_values[j];
        let y = &partial_dep[j];
        let (a, b) = ols_line(x, y);
        let effect = x
            .iter()
            .zip(y.iter())
            .map(|(vx, vy)| (a + b * *vx - *vy).abs())
            .sum::<f64>()
            / x.len() as f64;
        store.insert(j, effect);
    }
    store
}

fn get_pairwise_effect<F>(
    pairs: &[(usize, usize)],
    predictor: &F,
    x: &[Vec<f64>],
    num_values: usize,
    feature_values: &[Vec<f64>],
    partial_dep: &[Vec<f64>],
) -> BTreeMap<String, f64>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    let mut store = BTreeMap::new();
    for &(k, l) in pairs {
        let yk_centered = center(&partial_dep[k]);
        let yl_centered = center(&partial_dep[l]);
        let mut vals = Vec::with_capacity(num_values * num_values);

        for (ik, &xk) in feature_values[k].iter().enumerate() {
            for (il, &xl) in feature_values[l].iter().enumerate() {
                let mut x_mod = x.to_vec();
                for row in &mut x_mod {
                    row[k] = xk;
                    row[l] = xl;
                }
                let ykl = predictor(&x_mod).iter().sum::<f64>() / x.len() as f64;
                vals.push((ykl, yk_centered[ik], yl_centered[il]));
            }
        }

        let mean_ykl = vals.iter().map(|(v, _, _)| *v).sum::<f64>() / vals.len() as f64;
        let mut acc = 0.0;
        for (ykl, yk, yl) in vals {
            acc += (ykl - mean_ykl - yk - yl).abs();
        }
        store.insert(format!("({}, {})", k, l), acc / (num_values * num_values) as f64);
    }
    store
}

fn center(v: &[f64]) -> Vec<f64> {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| *x - m).collect()
}

fn normalize_usize_map(effect: &BTreeMap<usize, f64>) -> BTreeMap<usize, f64> {
    let sum: f64 = effect.values().sum();
    if sum == 0.0 {
        return effect.keys().map(|k| (*k, 0.0)).collect();
    }
    effect.iter().map(|(k, v)| (*k, *v / sum)).collect()
}

fn normalize_string_map(effect: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let sum: f64 = effect.values().sum();
    if sum == 0.0 {
        return effect.keys().map(|k| (k.clone(), 0.0)).collect();
    }
    effect.iter().map(|(k, v)| (k.clone(), *v / sum)).collect()
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

fn ols_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let mut cov = 0.0;
    let mut varx = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        cov += (*xi - mx) * (*yi - my);
        varx += (*xi - mx).powi(2);
    }
    let b = if varx > 0.0 { cov / varx } else { 0.0 };
    let a = my - b * mx;
    (a, b)
}
