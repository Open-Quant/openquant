#!/usr/bin/env python3
"""Hand-picked mutation audit behind docs/test-sensitivity-audit.md.

    python3 scripts/test-sensitivity/mutate.py            # every mutation
    python3 scripts/test-sensitivity/mutate.py hrp onc    # only ids starting with these prefixes

For each entry of MUTATIONS the script applies one textual change to a COPY of the workspace
(target/mutation-audit/src-copy, never the checkout itself), runs the listed test targets there
with that copy's own cargo target dir, and appends KILLED / SURVIVED to
docs/test-sensitivity-results.tsv. CARGO_TARGET_DIR is removed from the environment: sharing a
target dir between two source trees can run a stale library and invalidate the result.

ids starting with CTL- are controls: the function returns an obviously wrong constant. A control
that survives means the test target cannot see that function's value at all.

`after` lists the test targets added by the PR that introduced this script; when present the
mutation is run twice (before = old targets only, after = old + new).
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(ROOT, "target", "mutation-audit", "src-copy")
RESULTS = os.path.join(ROOT, "docs", "test-sensitivity-results.tsv")
K = "1234.5 + 0.0 * "

# (id, file under crates/openquant, old, new, test targets, extra targets added by this PR)
MUTATIONS = [
    # --- controls -------------------------------------------------------------------------
    ("CTL-backtest_statistics", "src/backtest_statistics.rs", "((mean - risk_free_rate) / std) * entries_per_year.sqrt()", K + "((mean - risk_free_rate) / std) * entries_per_year.sqrt()", "backtest_statistics", ""),
    ("CTL-bet_sizing", "src/bet_sizing.rs", "price_div * (w_param + price_div * price_div).powf(-0.5)", K + "price_div * (w_param + price_div * price_div).powf(-0.5)", "bet_sizing,ch10_snippets", ""),
    ("CTL-codependence", "src/codependence.rs", "Ok(cov / (var_x * var_y).sqrt())", "Ok(0.123 + 0.0 * cov / (var_x * var_y).sqrt())", "codependence", ""),
    ("CTL-combinatorial_optimization", "src/combinatorial_optimization.rs", "objective -= cfg.terminal_inventory_penalty * (terminal_diff as f64).powi(2);", "objective = 1234.5 + 0.0 * cfg.terminal_inventory_penalty * (terminal_diff as f64).powi(2);", "combinatorial_optimization", ""),
    ("CTL-cross_validation", "src/cross_validation.rs", "correct as f64 / y_test.len() as f64", K + "correct as f64 / y_test.len() as f64", "cross_validation", ""),
    ("CTL-data_structures", "src/data_structures.rs", "tick_count += 1;", "tick_count += 1000;", "data_structures_standard,data_structures_run_imbalance", ""),
    ("CTL-ef3m", "src/ef3m.rs", "    num / den", "    1234.5 + 0.0 * num / den", "ef3m", ""),
    ("CTL-ensemble_methods", "src/ensemble_methods.rs", "let variance = var_sum / n_samples;", "let variance = 1234.5 + 0.0 * var_sum / n_samples;", "ensemble_methods", ""),
    ("CTL-etf_trick", "src/etf_trick.rs", "h_t[j] = delever / denom;", "h_t[j] = 1234.5 + 0.0 * delever / denom;", "etf_trick", ""),
    ("CTL-futures_roll", "src/etf_trick.rs", "cum += g;", "cum += 1234.5 + 0.0 * g;", "futures_roll", ""),
    ("CTL-fast_ewma", "src/util/fast_ewma.rs", "out[i] = ewma_old / weight;", "out[i] = 1234.5 + 0.0 * ewma_old / weight;", "fast_ewma", ""),
    ("CTL-filters", "src/filters.rs", "let log_ret = (close[i] / close[i - 1]).ln();", "let log_ret = 1234.5 + 0.0 * (close[i] / close[i - 1]).ln();", "filters", ""),
    ("CTL-fingerprint", "src/fingerprint.rs", "let b = if varx > 0.0 { cov / varx } else { 0.0 };", "let b = 1234.5 + 0.0 * cov / varx;", "fingerprint", ""),
    ("CTL-fracdiff", "src/fracdiff.rs", "acc += *w * series[loc0 + k];", "acc += 1234.5 + 0.0 * *w * series[loc0 + k];", "fracdiff", "fracdiff_reference"),
    ("CTL-hpc_parallel", "src/hpc_parallel.rs", "PartitionStrategy::Linear => i * atom_count / molecules,", "PartitionStrategy::Linear => atom_count + 0 * i / molecules,", "hpc_parallel", ""),
    ("CTL-labeling", "src/labeling.rs", "let ret = (price / start_price - 1.0) * side;", "let ret = 1234.5 + 0.0 * (price / start_price - 1.0) * side;", "labeling", ""),
    ("CTL-microstructural_features", "src/microstructural_features.rs", "2.0 * (c.abs()).sqrt() }", "1234.5 + 0.0 * (c.abs()).sqrt() }", "microstructural_features", ""),
    ("CTL-risk_metrics", "src/risk_metrics.rs", "Ok(sorted[pos.min(n - 1)])", "Ok(1234.5 + 0.0 * sorted[pos.min(n - 1)])", "risk_metrics", "risk_metrics_reference"),
    ("CTL-sampling", "src/sampling.rs", "uniq_sum / count as f64", K + "uniq_sum / count as f64", "sampling", ""),
    ("CTL-sb_bagging", "src/sb_bagging.rs", "out[r] = if votes * 2 >= self.estimators.len() { 1 } else { 0 };", "out[r] = if votes * 2 >= self.estimators.len() { 1 } else { 1 };", "sb_bagging", ""),
    ("CTL-strategy_risk", "src/strategy_risk.rs", "let p = 0.5 * (1.0 + root);", "let p = 1234.5 + 0.0 * (1.0 + root);", "strategy_risk", ""),
    ("CTL-streaming_hpc", "src/streaming_hpc.rs", "Some(self.window_sum / self.window.len() as f64)", "Some(1234.5 + 0.0 * self.window_sum / self.window.len() as f64)", "streaming_hpc", ""),
    ("CTL-structural_breaks", "src/structural_breaks.rs", "stats.push(b_estimate / b_var.sqrt());", "stats.push(1234.5 + 0.0 * b_estimate / b_var.sqrt());", "structural_breaks", ""),
    ("CTL-synthetic_backtesting", "src/synthetic_backtesting.rs", "let phi = cov_xy / var_x;", "let phi = 1234.5 + 0.0 * cov_xy / var_x;", "synthetic_backtesting", ""),
    ("CTL-hrp", "src/hrp.rs", "let a = 1.0 - lv / (lv + rv + f64::EPSILON);", "let a = 1234.5 + 0.0 * lv / (lv + rv + f64::EPSILON);", "hrp", "hrp_reference"),
    ("CTL-hcaa", "src/hcaa.rs", "alloc_factor = alloc_factor.clamp(0.0, 1.0);", "alloc_factor = 1234.5 + 0.0 * alloc_factor.clamp(0.0, 1.0);", "hcaa", "hcaa_reference"),
    ("CTL-onc", "src/onc.rs", "(b - a) / a.max(b) };", "1234.5 + 0.0 * (b - a) / a.max(b) };", "onc,lib", "onc_reference"),
    ("CTL-backtesting_engine", "src/backtesting_engine.rs", "Ok((total * test_groups) / n_groups)", "Ok(12345 + 0 * (total * test_groups) / n_groups)", "backtesting_engine", "backtesting_engine_reference"),
    ("CTL-feature_importance", "src/feature_importance.rs", "let mean = if denom > 0.0 && means[j].is_finite() { means[j] / denom } else { 0.0 };", "let mean = 1234.5 + 0.0 * means[j] / denom;", "feature_importance", "feature_importance_reference"),
    ("CTL-hyperparameter_tuning", "src/hyperparameter_tuning.rs", "let accuracy = weighted_correct / sum_w;", "let accuracy = 1234.5 + 0.0 * weighted_correct / sum_w;", "hyperparameter_tuning", "hyperparameter_tuning_reference"),
    ("CTL-data_processing", "src/data_processing.rs", "gap_interval_count += 1;", "gap_interval_count += 1000;", "data_processing", ""),
    ("CTL-pipeline", "src/pipeline.rs", "equity *= 1.0 + strat_return;", "equity = 1234.5 + 0.0 * strat_return;", "lib", ""),
    ("CTL-sample_weights", "src/sample_weights.rs", "let scale = (weights.len() as f64) / total;", "let scale = 1234.5 + 0.0 * (weights.len() as f64) / total;", "sample_weights", ""),
    ("CTL-volatility", "src/util/volatility.rs", "(ret * ret) / (4.0 * 2.0f64.ln())", K + "(ret * ret) / (4.0 * 2.0f64.ln())", "volatility_features", ""),
    ("CTL-cla", "src/cla.rs", "*slot = (returns.column(c).sum() / rows as f64) * freq;", "*slot = 1234.5 + 0.0 * (returns.column(c).sum() / rows as f64) * freq;", "cla", ""),
    ("CTL-portfolio_optimization", "src/portfolio_optimization.rs", "s /= (rows - 1) as f64;", "s = 1234.5 + 0.0 * (rows - 1) as f64;", "portfolio_optimization", ""),
    # --- realistic mutations --------------------------------------------------------------
    ("backtest_statistics-1", "src/backtest_statistics.rs", "/ (returns.len() as f64 - 1.0);", "/ (returns.len() as f64);", "backtest_statistics", ""),
    ("backtest_statistics-2", "src/backtest_statistics.rs", "(1.0 - skewness * observed_sr", "(1.0 + skewness * observed_sr", "backtest_statistics", ""),
    ("backtest_statistics-3", "src/backtest_statistics.rs", "let adj = (hhi - 1.0 / n)", "let adj = (hhi + 1.0 / n)", "backtest_statistics", ""),
    ("bet_sizing-1", "src/bet_sizing.rs", "2.0 * norm.cdf(z) - 1.0", "2.0 * norm.cdf(z) - 0.9", "bet_sizing,ch10_snippets", ""),
    ("bet_sizing-2", "src/bet_sizing.rs", "if *s_ts <= *loc && (*loc < *end)", "if *s_ts < *loc && (*loc < *end)", "bet_sizing,ch10_snippets", ""),
    ("bet_sizing-3", "src/bet_sizing.rs", "(price_div * price_div) * ((1.0 / (m_bet_size * m_bet_size)) - 1.0)", "(price_div * price_div) * ((1.0 / (m_bet_size * m_bet_size)) + 1.0)", "bet_sizing,ch10_snippets", ""),
    ("codependence-1", "src/codependence.rs", "Ok((0.5 * (1.0 - corr_coef)).sqrt())", "Ok((0.5 * (1.0 + corr_coef)).sqrt())", "codependence", ""),
    ("codependence-2", "src/codependence.rs", "value -= p * p.ln();", "value -= p * p.log2();", "codependence", ""),
    ("combinatorial_optimization-1", "src/combinatorial_optimization.rs", "objective += directional_pnl - risk_penalty - impact_cost - fixed_cost;", "objective += directional_pnl - risk_penalty + impact_cost - fixed_cost;", "combinatorial_optimization", ""),
    ("combinatorial_optimization-2", "src/combinatorial_optimization.rs", "let risk_penalty = cfg.risk_aversion * inventory_after.powi(2);", "let risk_penalty = cfg.risk_aversion * inventory_after.abs();", "combinatorial_optimization", ""),
    ("cross_validation-1", "src/cross_validation.rs", "let after = (stop as isize + embargo)", "let after = (stop as isize + embargo - 1)", "cross_validation", ""),
    ("cross_validation-2", "src/cross_validation.rs", "let envelop = *s <= test_start && *e >= test_end;", "let envelop = false;", "cross_validation", ""),
    ("cross_validation-3", "src/cross_validation.rs", "2.0 * precision * recall / (precision + recall)", "precision * recall / (precision + recall)", "cross_validation", ""),
    ("data_structures-1", "src/data_structures.rs", "StandardBarType::Volume => volume >= threshold,", "StandardBarType::Volume => volume > threshold,", "data_structures_standard,data_structures_run_imbalance", ""),
    ("data_structures-2", "src/data_structures.rs", "if imbalance.abs() >= threshold {", "if imbalance >= threshold {", "data_structures_standard,data_structures_run_imbalance", ""),
    ("data_structures-3", "src/data_structures.rs", "dollar_value += trade.price * trade.volume;", "dollar_value += trade.price;", "data_structures_standard,data_structures_run_imbalance", ""),
    ("ef3m-1", "src/ef3m.rs", "let m_3 = p_1 * (3.0 * s_1.powi(2) * u_1 + u_1.powi(3))", "let m_3 = p_1 * (2.0 * s_1.powi(2) * u_1 + u_1.powi(3))", "ef3m", ""),
    ("ef3m-2", "src/ef3m.rs", "let mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1;", "let mu_1 = (m_1 + (1.0 - p_1) * mu_2) / p_1;", "ef3m", ""),
    ("ensemble_methods-1", "src/ensemble_methods.rs", "Ok(single_estimator_variance * (rho + (1.0 - rho) / n))", "Ok(single_estimator_variance * (rho + (1.0 + rho) / n))", "ensemble_methods", ""),
    ("ensemble_methods-2", "src/ensemble_methods.rs", "let noise = (mse - bias_sq - variance).max(0.0);", "let noise = (mse - bias_sq + variance).max(0.0);", "ensemble_methods", ""),
    ("ensemble_methods-3", "src/ensemble_methods.rs", "if votes * 2 >= per_model_predictions.len()", "if votes * 2 > per_model_predictions.len()", "ensemble_methods", ""),
    ("etf_trick-1", "src/etf_trick.rs", "k += h_prev[j] * rates.values[i][j] * (delta[j] + costs.values[i][j]);", "k += h_prev[j] * rates.values[i][j] * (delta[j] - costs.values[i][j]);", "etf_trick", ""),
    ("etf_trick-2", "src/etf_trick.rs", "let delever = weights[j] / abs_w_sum;", "let delever = weights[j];", "etf_trick", ""),
    ("futures_roll-1", "src/etf_trick.rs", "gaps[pos] = filtered[pos].open - filtered[pos - 1].close;", "gaps[pos] = filtered[pos].close - filtered[pos - 1].close;", "futures_roll", ""),
    ("fast_ewma-1", "src/util/fast_ewma.rs", "let alpha = 2.0 / (window as f64 + 1.0);", "let alpha = 2.0 / (window as f64);", "fast_ewma", ""),
    ("filters-1", "src/filters.rs", "if s_neg < -thresh {", "if s_neg <= -thresh * 1.05 {", "filters", ""),
    ("filters-2", "src/filters.rs", "/ (len - 1.0)", "/ len", "filters", ""),
    ("fingerprint-1", "src/fingerprint.rs", "acc += (ykl - mean_ykl - yk - yl).abs();", "acc += (ykl - mean_ykl - yk + yl).abs();", "fingerprint", ""),
    ("fingerprint-2", "src/fingerprint.rs", "let effect = x.iter().map(|v| (a + b * *v - y_mean).abs())", "let effect = x.iter().map(|v| (a + b * *v - y_mean).powi(2))", "fingerprint", ""),
    ("fracdiff-1", "src/fracdiff.rs", "let w = -weights[k - 1] * (diff_amt - k as f64 + 1.0) / k as f64;", "let w = -weights[k - 1] * (diff_amt - k as f64) / k as f64;", "fracdiff", "fracdiff_reference"),
    ("fracdiff-2", "src/fracdiff.rs", "let next = -weights[weights.len() - 1] * (diff_amt - k as f64 + 1.0) / k as f64;", "let next = weights[weights.len() - 1] * (diff_amt - k as f64 + 1.0) / k as f64;", "fracdiff", "fracdiff_reference"),
    ("hpc_parallel-1", "src/hpc_parallel.rs", "PartitionStrategy::Linear => i * atom_count / molecules,", "PartitionStrategy::Linear => i * atom_count / (molecules + 1),", "hpc_parallel", ""),
    ("hpc_parallel-2", "src/hpc_parallel.rs", "((atom_count as f64) * (i as f64 / molecules as f64).sqrt()).round() as usize", "((atom_count as f64) * (i as f64 / molecules as f64)).round() as usize", "hpc_parallel", ""),
    ("labeling-1", "src/labeling.rs", "let sl_level = if ev.sl > 0.0 { -ev.sl * ev.trgt }", "let sl_level = if ev.sl > 0.0 { -ev.sl * ev.trgt * 1.5 }", "labeling", ""),
    ("labeling-2", "src/labeling.rs", "let ret = (price / start_price - 1.0) * side;", "let ret = (price / start_price).ln() * side;", "labeling", ""),
    ("labeling-3", "src/labeling.rs", "if trgt <= config.min_ret {", "if trgt < config.min_ret * 0.5 {", "labeling", ""),
    ("microstructural_features-1", "src/microstructural_features.rs", "out[i] = s / (window as f64 - 1.0);", "out[i] = s / (window as f64);", "microstructural_features", ""),
    ("microstructural_features-2", "src/microstructural_features.rs", "let den = 3.0 - 2.0 * 2.0_f64.sqrt();", "let den = 3.0 + 2.0 * 2.0_f64.sqrt();", "microstructural_features", ""),
    ("risk_metrics-1", "src/risk_metrics.rs", "let tail: Vec<f64> = returns.iter().copied().filter(|v| *v < var).collect();", "let tail: Vec<f64> = returns.iter().copied().filter(|v| *v <= var).collect();", "risk_metrics", "risk_metrics_reference"),
    ("risk_metrics-2", "src/risk_metrics.rs", "let pos = (q * (n.saturating_sub(1) as f64)).ceil() as usize;", "let pos = (q * (n.saturating_sub(1) as f64)).floor() as usize;", "risk_metrics", "risk_metrics_reference"),
    ("risk_metrics-3", "src/risk_metrics.rs", "total += weights[i] * covariance[(i, j)] * weights[j];", "total += weights[i] * covariance[(i, j)];", "risk_metrics", "risk_metrics_reference"),
    ("sampling-1", "src/sampling.rs", "let new_el = val / (val + prev_concurrency[j]);", "let new_el = val / (1.0 + prev_concurrency[j] * 2.0);", "sampling", ""),
    ("sampling-2", "src/sampling.rs", "if *bar >= *start && *bar <= *end {", "if *bar >= *start && *bar < *end {", "sampling", ""),
    ("sb_bagging-1", "src/sb_bagging.rs", "positive_on_ge: rate_ge >= rate_lt,", "positive_on_ge: rate_ge < rate_lt,", "sb_bagging", ""),
    ("sb_bagging-2", "src/sb_bagging.rs", "(v * n_samples as f64) as usize", "(v * n_samples as f64 * 0.5) as usize", "sb_bagging", ""),
    ("strategy_risk-1", "src/strategy_risk.rs", "Ok((2.0 * precision - 1.0) / denom * annual_bet_frequency.sqrt())", "Ok((2.0 * precision - 1.0) / denom * annual_bet_frequency)", "strategy_risk", ""),
    ("strategy_risk-2", "src/strategy_risk.rs", "let b = (2.0 * n * payout.pi_minus - theta2 * d) * d;", "let b = (2.0 * n * payout.pi_minus + theta2 * d) * d;", "strategy_risk", ""),
    ("streaming_hpc-1", "src/streaming_hpc.rs", "self.current_bucket_abs_imbalance += (used_buy - used_sell).abs();", "self.current_bucket_abs_imbalance += (used_buy - used_sell);", "streaming_hpc", ""),
    ("streaming_hpc-2", "src/streaming_hpc.rs", "let toxicity = self.current_bucket_abs_imbalance / self.cfg.bucket_volume;", "let toxicity = self.current_bucket_abs_imbalance / self.current_bucket_volume.sqrt();", "streaming_hpc", ""),
    ("streaming_hpc-3", "src/streaming_hpc.rs", "self.sum_sq_counts += 2 * count_before + 1;", "self.sum_sq_counts += 2 * count_before + 2;", "streaming_hpc", ""),
    ("structural_breaks-1", "src/structural_breaks.rs", "max_s_n_critical_value = Some((4.6 + distance.ln()).sqrt());", "max_s_n_critical_value = Some((4.6 + distance).sqrt());", "structural_breaks", ""),
    ("structural_breaks-2", "src/structural_breaks.rs", "let sigma_sq_t = (1.0 / (index as f64 - 1.0)) * squared_diff_sum;", "let sigma_sq_t = (1.0 / (index as f64)) * squared_diff_sum;", "structural_breaks", ""),
    ("structural_breaks-3", "src/structural_breaks.rs", "let denom = rows as f64 - cols as f64;", "let denom = rows as f64;", "structural_breaks", ""),
    ("synthetic_backtesting-1", "src/synthetic_backtesting.rs", "let intercept = mean_y - phi * mean_x;", "let intercept = mean_y + phi * mean_x;", "synthetic_backtesting", ""),
    ("synthetic_backtesting-2", "src/synthetic_backtesting.rs", "let next = params.intercept + params.phi * prev + params.sigma * eps;", "let next = params.intercept + params.phi * prev + 2.0 * params.sigma * eps;", "synthetic_backtesting", ""),
    ("hrp-1", "src/hrp.rs", "let a = 1.0 - lv / (lv + rv + f64::EPSILON);", "let a = lv / (lv + rv + f64::EPSILON);", "hrp", "hrp_reference"),
    ("hrp-2", "src/hrp.rs", "d[(i, j)] = ((1.0 - c).max(0.0) / 2.0).sqrt();", "d[(i, j)] = ((1.0 + c).max(0.0) / 2.0).sqrt();", "hrp", "hrp_reference"),
    ("hrp-3", "src/hrp.rs", "out[(i, j)] *= 1.0 - a;", "out[(i, j)] *= a;", "hrp", "hrp_reference"),
    ("hrp-4", "src/hrp.rs", "inv_diag.push(1.0 / v);", "inv_diag.push(1.0 / v.sqrt());", "hrp", "hrp_reference"),
    ("hcaa-1", "src/hcaa.rs", "\"minimum_variance\" => 1.0 - left_var / (left_var + right_var + f64::EPSILON),", "\"minimum_variance\" => left_var / (left_var + right_var + f64::EPSILON),", "hcaa", "hcaa_reference"),
    ("hcaa-2", "src/hcaa.rs", "Ok(-tail.iter().sum::<f64>() / tail.len() as f64)", "Ok(tail.iter().sum::<f64>() / tail.len() as f64)", "hcaa", "hcaa_reference"),
    ("hcaa-3", "src/hcaa.rs", "Ok(mu / var.sqrt())", "Ok(mu / var)", "hcaa", "hcaa_reference"),
    ("hcaa-4", "src/hcaa.rs", "(0..cols).map(|c| returns.column(c).sum() / rows as f64 * 252.0).collect()", "(0..cols).map(|c| returns.column(c).sum() * 252.0).collect()", "hcaa", "hcaa_reference"),
    ("hcaa-5", "src/hcaa.rs", "1.0 - left_sd / (left_sd + right_sd + f64::EPSILON)", "left_sd / (left_sd + right_sd + f64::EPSILON)", "hcaa", "hcaa_reference"),
    ("hcaa-6", "src/hcaa.rs", "let dd = if peak > 0.0 { (peak - v) / peak } else { 0.0 };", "let dd = if peak > 0.0 { (peak - v) } else { 0.0 };", "hcaa", "hcaa_reference"),
    ("onc-1", "src/onc.rs", "(b - a) / a.max(b) };", "(b - a) / a.min(b) };", "onc,lib", "onc_reference"),
    ("onc-2", "src/onc.rs", "distance[(i, j)] = ((1.0 - c) / 2.0).sqrt();", "distance[(i, j)] = (1.0 - c) / 2.0;", "onc,lib", "onc_reference"),
    ("onc-3", "src/onc.rs", "        mean / std", "        mean / (std * std)", "onc,lib", "onc_reference"),
    ("onc-4", "src/onc.rs", "/ (own_members.len() - 1) as f64;", "/ own_members.len() as f64;", "onc,lib", "onc_reference"),
    ("backtesting_engine-1", "src/backtesting_engine.rs", "/ (n as f64 - 1.0)", "/ (n as f64)", "backtesting_engine", "backtesting_engine_reference"),
    ("backtesting_engine-2", "src/backtesting_engine.rs", "lhs.0 <= rhs.1 && rhs.0 <= lhs.1", "lhs.0 < rhs.1 && rhs.0 < lhs.1", "backtesting_engine", "backtesting_engine_reference"),
    ("backtesting_engine-3", "src/backtesting_engine.rs", "let stop = (*test_idx + embargo_width + 1).min(n_samples);", "let stop = (*test_idx + embargo_width).min(n_samples);", "backtesting_engine", "backtesting_engine_reference"),
    ("backtesting_engine-4", "src/backtesting_engine.rs", "{ mean / std * (n as f64).sqrt() }", "{ mean / std }", "backtesting_engine", "backtesting_engine_reference"),
    ("backtesting_engine-5", "src/backtesting_engine.rs", "path_returns.push(*r);", "path_returns.push(-*r);", "backtesting_engine", "backtesting_engine_reference"),
    ("feature_importance-1", "src/feature_importance.rs", "stderrs[j] = s * (per_tree_importances.len() as f64).powf(-0.5);", "stderrs[j] = s * (per_tree_importances.len() as f64).powf(-1.0);", "feature_importance", "feature_importance_reference"),
    ("feature_importance-2", "src/feature_importance.rs", "(base - perm) / (-perm)", "(base - perm) / perm", "feature_importance", "feature_importance_reference"),
    ("feature_importance-3", "src/feature_importance.rs", "all_eigs.push((evec[(r, c)] * eval[c]).abs());", "all_eigs.push(evec[(r, c)].abs());", "feature_importance", "feature_importance_reference"),
    ("feature_importance-4", "src/feature_importance.rs", "    (mean, var.sqrt())", "    (mean, var)", "feature_importance", "feature_importance_reference"),
    ("hyperparameter_tuning-1", "src/hyperparameter_tuning.rs", "let neg_log_loss = -(weighted_loss / sum_w);", "let neg_log_loss = weighted_loss / sum_w;", "hyperparameter_tuning", "hyperparameter_tuning_reference"),
    ("hyperparameter_tuning-2", "src/hyperparameter_tuning.rs", "Ok(draw.exp())", "Ok(draw.abs())", "hyperparameter_tuning", "hyperparameter_tuning_reference"),
    ("data_processing-1", "src/data_processing.rs", "t - pt > day_us", "t - pt >= day_us * 2", "data_processing", ""),
    ("data_processing-2", "src/data_processing.rs", "let step_us = interval_seconds * 1_000_000;", "let step_us = interval_seconds * 2_000_000;", "data_processing", ""),
    ("pipeline-1", "src/pipeline.rs", "let strat_return = timeline_signal[i - 1] * close_return;", "let strat_return = timeline_signal[i] * close_return;", "lib", ""),
    ("pipeline-2", "src/pipeline.rs", "let close_return = close[i] / close[i - 1] - 1.0;", "let close_return = (close[i] / close[i - 1]).ln();", "lib", ""),
    # --- out-of-scope modules (owned by #76): measured for the audit only -------------------
    ("sample_weights-1", "src/sample_weights.rs", "sum += ret / (*c as f64);", "sum += ret;", "sample_weights", ""),
    ("sample_weights-2", "src/sample_weights.rs", "denom += 1.0 / (*c as f64);", "denom += 1.0;", "sample_weights", ""),
    ("volatility-1", "src/util/volatility.rs", "let c = 2.0 * 2.0f64.ln() - 1.0;", "let c = 2.0 * 2.0f64.ln() + 1.0;", "volatility_features", ""),
    ("volatility-2", "src/util/volatility.rs", "let k = 0.34 / (1.34 +", "let k = 0.34 / (1.0 +", "volatility_features", ""),
    ("cla-1", "src/cla.rs", "ema = alpha * returns[(r, c)] + (1.0 - alpha) * ema;", "ema = alpha * returns[(r, c)] + (1.0 + alpha) * ema;", "cla", ""),
    ("cla-2", "src/cla.rs", "let sigma = quad_risk(&self.cov_matrix, weights).sqrt();", "let sigma = quad_risk(&self.cov_matrix, weights);", "cla", ""),
    ("portfolio_optimization-1", "src/portfolio_optimization.rs", "out[(r - 1, c)] = (prices[(r, c)] / prev).ln();", "out[(r - 1, c)] = prices[(r, c)] / prev - 1.0;", "portfolio_optimization", ""),
    ("portfolio_optimization-2", "src/portfolio_optimization.rs", "expected[c] = (num / denom) * freq;", "expected[c] = num * freq;", "portfolio_optimization", ""),
]


def sync_copy():
    os.makedirs(WORK, exist_ok=True)
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "clippy.toml", "rustfmt.toml"):
        shutil.copy2(os.path.join(ROOT, name), os.path.join(WORK, name))
    for d in ("crates", "vendor", "tests"):
        dst = os.path.join(WORK, d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(ROOT, d), dst)


def run(path, src, old, new, targets):
    assert src.count(old) >= 1, f"mutation site not found: {old!r}"
    env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
    args = ["cargo", "test", "-p", "openquant"]
    for t in targets:
        args += ["--lib"] if t == "lib" else ["--test", t]
    open(path, "w").write(src.replace(old, new, 1))
    try:
        r = subprocess.run(args, cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
    finally:
        open(path, "w").write(src)
    out = r.stdout + r.stderr
    if "could not compile" in out:
        return "COMPILE_ERROR"
    if "Compiling openquant" not in out:
        return "INVALID(no rebuild)"
    if r.returncode == 0:
        return "SURVIVED"
    failed = [l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.rstrip().endswith("FAILED")]
    return "KILLED by " + ",".join(failed[:4]) + (f" (+{len(failed) - 4})" if len(failed) > 4 else "")


def main():
    prefixes = sys.argv[1:]
    sync_copy()
    with open(RESULTS, "a") as log:
        for mid, f, old, new, tests, after in MUTATIONS:
            if prefixes and not any(mid.startswith(p) or mid.startswith("CTL-" + p) for p in prefixes):
                continue
            path = os.path.join(WORK, "crates", "openquant", f)
            src = open(path).read()
            phases = [("before", tests.split(","))]
            if after:
                phases.append(("after", tests.split(",") + after.split(",")))
            for phase, targets in phases:
                res = run(path, src, old, new, targets)
                line = "\t".join([mid, phase, f, f"`{old}` -> `{new}`", ",".join(targets), res])
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()


if __name__ == "__main__":
    main()
