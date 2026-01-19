use chrono::NaiveDateTime;
use statrs::distribution::{ContinuousCDF, Normal};

const EULER_GAMMA: f64 = 0.5772156649015329_f64;

pub fn timing_of_flattening_and_flips(
    target_positions: &[(NaiveDateTime, f64)],
) -> Vec<NaiveDateTime> {
    let mut flattenings = Vec::new();
    let mut flips = Vec::new();
    for i in 1..target_positions.len() {
        let prev = target_positions[i - 1].1;
        let curr = target_positions[i].1;
        if curr == 0.0 && prev != 0.0 {
            flattenings.push(target_positions[i].0);
        }
        let mult = curr * prev;
        if mult < 0.0 {
            flips.push(target_positions[i].0);
        }
    }
    let mut res = flattenings;
    res.extend(flips);
    res.sort();
    res.dedup();
    if let Some(last) = target_positions.last() {
        if !res.contains(&last.0) {
            res.push(last.0);
        }
    }
    res
}

pub fn average_holding_period(target_positions: &[(NaiveDateTime, f64)]) -> Option<f64> {
    if target_positions.is_empty() {
        return None;
    }
    let mut holding: Vec<(f64, f64)> = Vec::new(); // (holding_time_days, weight)
    let mut entry_time = 0.0;
    let time_since_start: Vec<f64> = target_positions
        .iter()
        .map(|(ts, _)| (*ts - target_positions[0].0).num_seconds() as f64 / 86_400.0)
        .collect();
    let mut position_diff: Vec<f64> = target_positions.iter().map(|(_, v)| *v).collect();
    for i in (1..position_diff.len()).rev() {
        position_diff[i] -= position_diff[i - 1];
    }
    for i in 1..target_positions.len() {
        let prev_pos = target_positions[i - 1].1;
        let diff = position_diff[i];
        let curr_pos = target_positions[i].1;
        if diff * prev_pos >= 0.0 && curr_pos != 0.0 {
            entry_time = (entry_time * prev_pos + time_since_start[i] * diff) / curr_pos;
        }
        if diff * prev_pos < 0.0 {
            let hold_time = time_since_start[i] - entry_time;
            if curr_pos * prev_pos < 0.0 {
                let weight = prev_pos.abs();
                holding.push((hold_time, weight));
                entry_time = time_since_start[i];
            } else {
                let weight = diff.abs();
                holding.push((hold_time, weight));
            }
        }
    }
    let total_w: f64 = holding.iter().map(|(_, w)| *w).sum();
    if total_w > 0.0 {
        let num: f64 = holding.iter().map(|(h, w)| h * w).sum();
        Some(num / total_w)
    } else {
        None
    }
}

pub fn bets_concentration(returns: &[f64]) -> Option<f64> {
    if returns.len() <= 2 {
        return None;
    }
    let sum: f64 = returns.iter().sum();
    if sum == 0.0 {
        return None;
    }
    let weights: Vec<f64> = returns.iter().map(|r| r / sum).collect();
    let hhi: f64 = weights.iter().map(|w| w * w).sum();
    let n = returns.len() as f64;
    let adj = (hhi - 1.0 / n) / (1.0 - 1.0 / n);
    Some(adj)
}

pub fn all_bets_concentration(
    returns: &[(NaiveDateTime, f64)],
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let positives: Vec<f64> = returns.iter().filter(|(_, r)| *r >= 0.0).map(|(_, r)| *r).collect();
    let negatives: Vec<f64> = returns.iter().filter(|(_, r)| *r < 0.0).map(|(_, r)| *r).collect();
    let pos = bets_concentration(&positives);
    let neg = bets_concentration(&negatives);
    // time grouping by day including gaps between first and last date (zeros matter)
    let mut per_day: std::collections::HashMap<chrono::NaiveDate, usize> =
        std::collections::HashMap::new();
    for (ts, _) in returns {
        *per_day.entry(ts.date()).or_insert(0) += 1;
    }
    let time = if returns.is_empty() {
        None
    } else {
        let start = returns.first().unwrap().0.date();
        let end = returns.last().unwrap().0.date();
        let mut counts: Vec<f64> = Vec::new();
        let mut day = start;
        while day <= end {
            let cnt = per_day.get(&day).copied().unwrap_or(0) as f64;
            counts.push(cnt);
            day = day.succ_opt().unwrap();
        }
        bets_concentration(&counts)
    };
    (pos, neg, time)
}

pub fn drawdown_and_time_under_water(
    returns: &[(NaiveDateTime, f64)],
    dollars: bool,
) -> (Vec<f64>, Vec<f64>) {
    if returns.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Track high-water-mark segments and their minima (mirrors pandas grouping in Python version)
    let mut hwms: Vec<f64> = vec![returns[0].1];
    let mut hwm_times: Vec<NaiveDateTime> = vec![returns[0].0];
    let mut segment_min: Vec<f64> = vec![returns[0].1];

    for &(ts, val) in returns.iter().skip(1) {
        let current_hwm = *hwms.last().unwrap();
        if val > current_hwm {
            // start new HWM segment
            hwms.push(val);
            hwm_times.push(ts);
            segment_min.push(val);
        } else {
            let last_min = segment_min.last_mut().unwrap();
            if val < *last_min {
                *last_min = val;
            }
        }
    }

    // Compute drawdowns only for segments that actually dipped below HWM
    let mut drawdowns = Vec::new();
    let mut dd_times = Vec::new();
    for i in 0..hwms.len() {
        if segment_min[i] < hwms[i] {
            let dd =
                if dollars { hwms[i] - segment_min[i] } else { 1.0 - segment_min[i] / hwms[i] };
            drawdowns.push(dd);
            dd_times.push(hwm_times[i]);
        }
    }

    // Time under water between consecutive HWMs that had drawdowns, plus last interval to series end
    let mut tuw = Vec::new();
    for i in 0..dd_times.len() {
        let start = dd_times[i];
        let end = if i + 1 < dd_times.len() { dd_times[i + 1] } else { returns.last().unwrap().0 };
        let years = (end - start).num_seconds() as f64 / (365.25 * 24.0 * 3600.0);
        tuw.push(years);
    }

    (drawdowns, tuw)
}

pub fn sharpe_ratio(returns: &[f64], entries_per_year: f64, risk_free_rate: f64) -> f64 {
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() as f64 - 1.0);
    let std = var.sqrt();
    ((mean - risk_free_rate) / std) * entries_per_year.sqrt()
}

pub fn information_ratio(returns: &[f64], benchmark: f64, entries_per_year: f64) -> f64 {
    let excess: Vec<f64> = returns.iter().map(|r| r - benchmark).collect();
    sharpe_ratio(&excess, entries_per_year, 0.0)
}

pub fn probabilistic_sharpe_ratio(
    observed_sr: f64,
    benchmark_sr: f64,
    number_of_returns: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    let z = ((observed_sr - benchmark_sr) * (number_of_returns as f64 - 1.0).sqrt())
        / (1.0 - skewness * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr * observed_sr)
            .sqrt();
    let norm = Normal::new(0.0, 1.0).unwrap();
    norm.cdf(z)
}

pub fn deflated_sharpe_ratio(
    observed_sr: f64,
    sr_estimates: &[f64],
    number_of_returns: usize,
    skewness: f64,
    kurtosis: f64,
    estimates_param: bool,
    benchmark_out: bool,
) -> f64 {
    let benchmark_sr = if estimates_param {
        let sd = sr_estimates[0];
        let n = sr_estimates[1];
        let norm = Normal::new(0.0, 1.0).unwrap();
        sd * ((1.0 - EULER_GAMMA) * norm.inverse_cdf(1.0 - 1.0 / n)
            + EULER_GAMMA * norm.inverse_cdf(1.0 - 1.0 / n * (-1.0f64).exp()))
    } else {
        let sd = {
            let mean = sr_estimates.iter().sum::<f64>() / sr_estimates.len() as f64;
            (sr_estimates.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / sr_estimates.len() as f64)
                .sqrt()
        };
        let n = sr_estimates.len() as f64;
        let norm = Normal::new(0.0, 1.0).unwrap();
        sd * ((1.0 - EULER_GAMMA) * norm.inverse_cdf(1.0 - 1.0 / n)
            + EULER_GAMMA * norm.inverse_cdf(1.0 - 1.0 / n * (-1.0f64).exp()))
    };

    if benchmark_out {
        return benchmark_sr;
    }

    probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns, skewness, kurtosis)
}

pub fn minimum_track_record_length(
    observed_sr: f64,
    benchmark_sr: f64,
    skewness: f64,
    kurtosis: f64,
    alpha: f64,
) -> f64 {
    let norm = Normal::new(0.0, 1.0).unwrap();
    let z = norm.inverse_cdf(1.0 - alpha);
    1.0 + (1.0 - skewness * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr * observed_sr)
        * (z / (observed_sr - benchmark_sr)).powi(2)
}
