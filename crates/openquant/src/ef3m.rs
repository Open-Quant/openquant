use rand::Rng;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone)]
pub struct M2N {
    pub moments: Vec<f64>,
    pub epsilon: f64,
    pub factor: f64,
    pub n_runs: usize,
    pub variant: usize,
    pub max_iter: usize,
    pub num_workers: isize,
    pub new_moments: Vec<f64>,
    pub parameters: Vec<f64>,
    pub error: f64,
}

#[derive(Debug, Clone)]
pub struct FitResultRow {
    pub mu_1: f64,
    pub mu_2: f64,
    pub sigma_1: f64,
    pub sigma_2: f64,
    pub p_1: f64,
    pub error: f64,
}

fn comb(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k_eff = k.min(n - k);
    let mut num = 1.0;
    let mut den = 1.0;
    for i in 0..k_eff {
        num *= (n - i) as f64;
        den *= (i + 1) as f64;
    }
    num / den
}

fn round_to_5(x: f64) -> f64 {
    (x * 100_000.0).round() / 100_000.0
}

impl M2N {
    pub fn new(
        moments: Vec<f64>,
        epsilon: f64,
        factor: f64,
        n_runs: usize,
        variant: usize,
        max_iter: usize,
        num_workers: isize,
    ) -> Self {
        let error = moments.iter().map(|m| m * m).sum();
        Self {
            moments,
            epsilon,
            factor,
            n_runs,
            variant,
            max_iter,
            num_workers,
            new_moments: vec![0.0; 5],
            parameters: vec![0.0; 5],
            error,
        }
    }

    pub fn with_defaults(moments: Vec<f64>) -> Self {
        Self::new(moments, 1e-5, 5.0, 1, 1, 100_000, -1)
    }

    pub fn get_moments(&mut self, parameters: &[f64], return_result: bool) -> Option<Vec<f64>> {
        let u_1 = parameters[0];
        let u_2 = parameters[1];
        let s_1 = parameters[2];
        let s_2 = parameters[3];
        let p_1 = parameters[4];
        let p_2 = 1.0 - p_1;

        let m_1 = p_1 * u_1 + p_2 * u_2;
        let m_2 = p_1 * (s_1.powi(2) + u_1.powi(2)) + p_2 * (s_2.powi(2) + u_2.powi(2));
        let m_3 = p_1 * (3.0 * s_1.powi(2) * u_1 + u_1.powi(3))
            + p_2 * (3.0 * s_2.powi(2) * u_2 + u_2.powi(3));
        let m_4 = p_1 * (3.0 * s_1.powi(4) + 6.0 * s_1.powi(2) * u_1.powi(2) + u_1.powi(4))
            + p_2 * (3.0 * s_2.powi(4) + 6.0 * s_2.powi(2) * u_2.powi(2) + u_2.powi(4));
        let m_5 = p_1 * (15.0 * s_1.powi(4) * u_1 + 10.0 * s_1.powi(2) * u_1.powi(3) + u_1.powi(5))
            + p_2 * (15.0 * s_2.powi(4) * u_2 + 10.0 * s_2.powi(2) * u_2.powi(3) + u_2.powi(5));
        let out = vec![m_1, m_2, m_3, m_4, m_5];

        if return_result {
            Some(out)
        } else {
            self.new_moments = out;
            None
        }
    }

    pub fn iter_4(&self, mu_2: f64, p_1: f64) -> Vec<f64> {
        let m_1 = self.moments[0];
        let m_2 = self.moments[1];
        let m_3 = self.moments[2];
        let m_4 = self.moments[3];

        let mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1;
        let den_24 = 3.0 * (1.0 - p_1) * (mu_2 - mu_1);
        if den_24 == 0.0 {
            return vec![];
        }
        let sigma_2_squared = (m_3 + 2.0 * p_1 * mu_1.powi(3) + (p_1 - 1.0) * mu_2.powi(3)
            - 3.0 * mu_1 * (m_2 + mu_2.powi(2) * (p_1 - 1.0)))
            / den_24;
        if sigma_2_squared < 0.0 {
            return vec![];
        }
        let sigma_2 = sigma_2_squared.sqrt();

        let sigma_1_squared =
            ((m_2 - sigma_2.powi(2) - mu_2.powi(2)) / p_1) + sigma_2.powi(2) + mu_2.powi(2)
                - mu_1.powi(2);
        if sigma_1_squared < 0.0 {
            return vec![];
        }
        let sigma_1 = sigma_1_squared.sqrt();

        let p_1_deno = 3.0 * (sigma_1.powi(4) - sigma_2.powi(4))
            + 6.0 * (sigma_1.powi(2) * mu_1.powi(2) - sigma_2.powi(2) * mu_2.powi(2))
            + mu_1.powi(4)
            - mu_2.powi(4);
        if p_1_deno == 0.0 {
            return vec![];
        }
        let p_1_new =
            (m_4 - 3.0 * sigma_2.powi(4) - 6.0 * sigma_2.powi(2) * mu_2.powi(2) - mu_2.powi(4))
                / p_1_deno;
        if !(0.0..=1.0).contains(&p_1_new) {
            return vec![];
        }

        vec![mu_1, mu_2, sigma_1, sigma_2, p_1_new]
    }

    pub fn iter_5(&self, mu_2: f64, p_1: f64) -> Vec<f64> {
        let m_1 = self.moments[0];
        let m_2 = self.moments[1];
        let m_3 = self.moments[2];
        let m_4 = self.moments[3];
        let m_5 = self.moments[4];

        let mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1;
        let den_24 = 3.0 * (1.0 - p_1) * (mu_2 - mu_1);
        if den_24 == 0.0 {
            return vec![];
        }
        let sigma_2_squared = (m_3 + 2.0 * p_1 * mu_1.powi(3) + (p_1 - 1.0) * mu_2.powi(3)
            - 3.0 * mu_1 * (m_2 + mu_2.powi(2) * (p_1 - 1.0)))
            / den_24;
        if sigma_2_squared < 0.0 {
            return vec![];
        }
        let sigma_2 = sigma_2_squared.sqrt();

        let sigma_1_squared =
            ((m_2 - sigma_2.powi(2) - mu_2.powi(2)) / p_1) + sigma_2.powi(2) + mu_2.powi(2)
                - mu_1.powi(2);
        if sigma_1_squared < 0.0 {
            return vec![];
        }
        let sigma_1 = sigma_1_squared.sqrt();

        if (1.0 - p_1) < 1e-4 {
            return vec![];
        }
        let a_1_squared = 6.0 * sigma_2.powi(4)
            + (m_4
                - p_1
                    * (3.0 * sigma_1.powi(4)
                        + 6.0 * sigma_1.powi(2) * mu_1.powi(2)
                        + mu_1.powi(4)))
                / (1.0 - p_1);
        if a_1_squared < 0.0 {
            return vec![];
        }
        let a_1 = a_1_squared.sqrt();
        let mu_2_squared = a_1 - 3.0 * sigma_2.powi(2);
        if !mu_2_squared.is_finite() || mu_2_squared < 0.0 {
            return vec![];
        }
        let mu_2_new = mu_2_squared.sqrt();

        let a_2 =
            15.0 * sigma_1.powi(4) * mu_1 + 10.0 * sigma_1.powi(2) * mu_1.powi(3) + mu_1.powi(5);
        let b_2 = 15.0 * sigma_2.powi(4) * mu_2_new
            + 10.0 * sigma_2.powi(2) * mu_2_new.powi(3)
            + mu_2_new.powi(5);
        if (a_2 - b_2) == 0.0 {
            return vec![];
        }
        let p_1_new = (m_5 - b_2) / (a_2 - b_2);
        if !(0.0..=1.0).contains(&p_1_new) {
            return vec![];
        }

        vec![mu_1, mu_2_new, sigma_1, sigma_2, p_1_new]
    }

    pub fn fit(&mut self, mut mu_2: f64) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut p_1 = rng.gen_range(0.0..1.0);
        let mut num_iter = 0usize;

        loop {
            num_iter += 1;
            let parameters_new = match self.variant {
                1 => self.iter_4(mu_2, p_1),
                2 => self.iter_5(mu_2, p_1),
                _ => return Err("Value of argument 'variant' must be either 1 or 2.".to_string()),
            };
            if parameters_new.is_empty() {
                return Ok(());
            }

            let parameters = parameters_new.clone();
            let _ = self.get_moments(&parameters, false);
            let error: f64 = self
                .moments
                .iter()
                .zip(self.new_moments.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if error < self.error {
                self.parameters = parameters.clone();
                self.error = error;
            }

            if (p_1 - parameters[4]).abs() < self.epsilon {
                self.parameters = parameters;
                break;
            }
            if num_iter > self.max_iter {
                return Ok(());
            }
            p_1 = parameters[4];
            mu_2 = parameters[1];
        }
        Ok(())
    }

    pub fn single_fit_loop(&mut self, epsilon_override: Option<f64>) -> Vec<FitResultRow> {
        if let Some(eps) = epsilon_override {
            if eps > 0.0 {
                self.epsilon = eps;
            }
        }
        self.parameters = vec![0.0; 5];
        self.error = self.moments.iter().map(|m| m * m).sum();

        let std_dev = centered_moment(&self.moments, 2).sqrt();
        let upper = (1.0 / self.epsilon).max(1.0) as usize;
        let mut err_min = self.error;
        let mut best: Option<FitResultRow> = None;

        for i in 1..upper {
            let mu_2_i = i as f64 * self.epsilon * self.factor * std_dev + self.moments[0];
            let _ = self.fit(mu_2_i);
            if self.error < err_min {
                err_min = self.error;
                best = Some(FitResultRow {
                    mu_1: self.parameters[0],
                    mu_2: self.parameters[1],
                    sigma_1: self.parameters[2],
                    sigma_2: self.parameters[3],
                    p_1: self.parameters[4],
                    error: err_min,
                });
            }
        }

        best.into_iter().collect()
    }

    pub fn mp_fit(&self) -> Vec<FitResultRow> {
        let mut out = Vec::new();
        for _ in 0..self.n_runs {
            let mut worker = self.clone();
            out.extend(worker.single_fit_loop(Some(worker.epsilon)));
        }
        out
    }
}

pub fn centered_moment(moments: &[f64], order: usize) -> f64 {
    let mut moment_c = 0.0;
    for j in 0..=order {
        let combin = comb(order, j);
        let a_1 = if j == order { 1.0 } else { moments[order - j - 1] };
        moment_c += (-1.0f64).powi(j as i32) * combin * moments[0].powi(j as i32) * a_1;
    }
    moment_c
}

pub fn raw_moment(central_moments: &[f64], dist_mean: f64) -> Vec<f64> {
    let mut raw_moments = vec![dist_mean];
    let mut central = vec![1.0];
    central.extend_from_slice(central_moments);
    for n_i in 2..central.len() {
        let mut moment_n = 0.0;
        for (k, ck) in central.iter().take(n_i + 1).enumerate() {
            moment_n += comb(n_i, k) * ck * dist_mean.powi((n_i - k) as i32);
        }
        raw_moments.push(moment_n);
    }
    raw_moments
}

pub fn most_likely_parameters(
    data: &[FitResultRow],
    ignore_columns: Option<&[&str]>,
    res: usize,
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if data.is_empty() {
        return out;
    }
    let ignored: HashSet<&str> = ignore_columns.unwrap_or(&["error"]).iter().copied().collect();

    let columns: [(&str, Vec<f64>); 6] = [
        ("mu_1", data.iter().map(|r| r.mu_1).collect()),
        ("mu_2", data.iter().map(|r| r.mu_2).collect()),
        ("sigma_1", data.iter().map(|r| r.sigma_1).collect()),
        ("sigma_2", data.iter().map(|r| r.sigma_2).collect()),
        ("p_1", data.iter().map(|r| r.p_1).collect()),
        ("error", data.iter().map(|r| r.error).collect()),
    ];

    for (name, vals) in columns {
        if ignored.contains(name) {
            continue;
        }
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() < 1e-15 {
            out.insert(name.to_string(), round_to_5(min));
            continue;
        }

        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n.max(1.0);
        let std = var.sqrt().max(1e-12);
        let h = (std * n.powf(-1.0 / 5.0)).max(1e-6);

        let steps = res.max(10);
        let dx = (max - min) / (steps as f64 - 1.0);
        let mut best_x = min;
        let mut best_y = f64::NEG_INFINITY;
        for i in 0..steps {
            let x = min + dx * i as f64;
            let y = vals
                .iter()
                .map(|v| {
                    let u = (x - v) / h;
                    (-0.5 * u * u).exp()
                })
                .sum::<f64>()
                / (n * h * (2.0 * std::f64::consts::PI).sqrt());
            if y > best_y {
                best_y = y;
                best_x = x;
            }
        }
        out.insert(name.to_string(), round_to_5(best_x));
    }

    out
}
