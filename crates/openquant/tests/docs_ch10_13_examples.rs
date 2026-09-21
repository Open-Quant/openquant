//! The Rust examples on the backtest_statistics, backtesting_engine and synthetic_backtesting
//! docs pages, run as tests. `check:examples` only compiles page snippets; these execute them.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn backtest_statistics_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::backtest_statistics::{
        bets_concentration, deflated_sharpe_ratio, minimum_track_record_length,
        probabilistic_sharpe_ratio, sharpe_ratio,
    };

    // Per-period Sharpe 0.1 over 500 normal returns: confident it is positive...
    let psr = probabilistic_sharpe_ratio(0.1, 0.0, 500, 0.0, 3.0);
    assert!((psr - 0.9871).abs() < 1e-4);
    // ...less so with negative skew and fat tails, and less again after 50 trials.
    assert!(probabilistic_sharpe_ratio(0.1, 0.0, 500, -2.0, 10.0) < psr);
    let hurdle = deflated_sharpe_ratio(0.1, &[0.05, 50.0], 500, 0.0, 3.0, true, true)?;
    assert!((hurdle - 0.1138).abs() < 1e-4);
    assert!(deflated_sharpe_ratio(0.1, &[0.05, 50.0], 500, 0.0, 3.0, true, false)? < 0.5);

    // Observations needed to be 95% confident that a per-period Sharpe of 0.1 beats zero.
    let min_trl = minimum_track_record_length(0.1, 0.0, 0.0, 3.0, 0.05)?;
    assert!((min_trl - 272.9).abs() < 0.1);

    // Equal bets are not concentrated; one dominant bet is.
    assert!(bets_concentration(&[1.0, 1.0, 1.0, 1.0]).unwrap().abs() < 1e-12);
    assert!(bets_concentration(&[97.0, 1.0, 1.0, 1.0]).unwrap() > 0.9);

    // Annualising multiplies by the square root of the periods per year.
    let r = [0.01, -0.005, 0.007, 0.002, -0.001];
    assert!(
        (sharpe_ratio(&r, 252.0, 0.0) / sharpe_ratio(&r, 1.0, 0.0) - 252f64.sqrt()).abs() < 1e-9
    );
    Ok(())
}

#[test]
fn backtesting_engine_page() -> Result<(), Box<dyn std::error::Error>> {
    use chrono::{Duration, NaiveDate};
    use openquant::backtesting_engine::{
        cpcv_path_count, run_cpcv, run_walk_forward, BacktestData, BacktestError,
        BacktestRunConfig, BacktestSafeguards, CpcvConfig, SplitDefinition, WalkForwardConfig,
    };

    // Reproducible noise in [-1, 1) from a hash, so the example needs no RNG.
    fn noise(i: usize, salt: u64) -> f64 {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
        h ^= h >> 31;
        h = h.wrapping_mul(0x7FB5_D329_728E_A185);
        h ^= h >> 27;
        (h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    // 240 daily returns, each carrying over a quarter of the one before. Labels span 5 days.
    let n = 240;
    let mut asset = vec![0.0; n];
    for i in 1..n {
        asset[i] = 0.25 * asset[i - 1] + 0.01 * noise(i, 5);
    }
    let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    let label_spans: Vec<_> =
        (0..n as i64).map(|i| (open + Duration::days(i), open + Duration::days(i + 4))).collect();
    let data = BacktestData { returns: asset.clone(), label_spans };

    let note = |s: &str| s.to_string();
    let run = BacktestRunConfig {
        mode_provenance: note("docs example: one rule family, fixed before the run"),
        trials_count: 1,
        safeguards: BacktestSafeguards {
            survivorship_bias_control: note("synthetic single series"),
            look_ahead_control: note("lookback chosen on training rows only"),
            data_mining_control: note("two candidate lookbacks, declared in advance"),
            cost_assumption: note("none modelled"),
            multiple_testing_control: note("single trial"),
        },
    };

    // Follow the sign of the last `lookback` returns.
    let rule = |lookback: usize, i: usize| -> f64 {
        if i < lookback {
            return 0.0;
        }
        asset[i - lookback..i].iter().sum::<f64>().signum() * asset[i]
    };
    let evaluator = |split: &SplitDefinition| -> Result<Vec<f64>, BacktestError> {
        let earned = |l: usize| split.train_indices.iter().map(|&i| rule(l, i)).sum::<f64>();
        let chosen = if earned(1) >= earned(10) { 1 } else { 10 };
        Ok(split.test_indices.iter().map(|&i| rule(chosen, i)).collect())
    };

    let walk = run_walk_forward(
        &data,
        &run,
        &WalkForwardConfig { min_train_size: 80, test_size: 40, step_size: 40, pct_embargo: 0.0 },
        evaluator,
    )?;
    assert_eq!(walk.folds.len(), 4); // one path, covering only samples 80-239

    let cpcv = run_cpcv(
        &data,
        &run,
        &CpcvConfig { n_groups: 6, test_groups: 2, pct_embargo: 0.0 },
        evaluator,
    )?;
    assert_eq!((cpcv.splits.len(), cpcv.path_count), (15, cpcv_path_count(6, 2)?));
    assert_eq!(cpcv.path_count, 5);
    assert!(cpcv.path_distribution.iter().all(|p| p.observations == 240));
    // Path 0 takes groups 0 and 1 from split 0, then one group from each of splits 1 to 4.
    assert_eq!(cpcv.path_assignments[0].split_for_group, vec![0, 0, 1, 2, 3, 4]);

    // Two adjacent test groups at the start have one boundary with the training set: 4 purged.
    // Two separated interior groups have four: 16 purged.
    assert_eq!((cpcv.splits[0].test_groups.clone(), cpcv.splits[0].purged_count), (vec![0, 1], 4));
    assert_eq!((cpcv.splits[7].test_groups.clone(), cpcv.splits[7].purged_count), (vec![1, 4], 16));

    let t_stats: Vec<f64> = cpcv.path_distribution.iter().map(|p| p.sharpe).collect();
    let (low, high) = (
        t_stats.iter().cloned().fold(f64::MAX, f64::min),
        t_stats.iter().cloned().fold(f64::MIN, f64::max),
    );
    assert!((low - 2.40).abs() < 0.01 && (high - 3.20).abs() < 0.01);
    Ok(())
}

#[test]
fn synthetic_backtesting_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::synthetic_backtesting::{
        calibrate_ou_params, evaluate_rule_on_paths, generate_ou_paths, OuProcessParams,
        TradingRule,
    };

    // An exact AR(1) recursion with no noise term would have sigma = 0 and be refused, so
    // perturb it slightly: phi = 0.8 around 50.
    let mut prices = vec![40.0];
    for i in 1..400 {
        let wobble = if i % 2 == 0 { 0.05 } else { -0.05 };
        prices.push(0.2 * 50.0 + 0.8 * prices[i - 1] + wobble);
    }
    let fitted = calibrate_ou_params(&prices)?;
    assert!((fitted.phi - 0.8).abs() < 0.01 && (fitted.equilibrium - 50.0).abs() < 0.01);
    assert!(fitted.stationary);

    // Simulation is reproducible for a seed, and every path starts at the entry price.
    let params = OuProcessParams {
        phi: 0.9,
        intercept: 10.0,
        equilibrium: 100.0,
        sigma: 1.0,
        r_squared: 0.0,
        stationary: true,
    };
    let paths = generate_ou_paths(params, 97.0, 2_000, 60, 11)?;
    assert_eq!(paths, generate_ou_paths(params, 97.0, 2_000, 60, 11)?);
    assert!(paths.iter().all(|p| p.len() == 60 && p[0] == 97.0));

    // Entered three points below equilibrium, a wide stop beats a tight one.
    let wide = evaluate_rule_on_paths(
        &paths,
        TradingRule { profit_taking: 2.0, stop_loss: 8.0 },
        59,
        1.0,
    )?;
    let tight = evaluate_rule_on_paths(
        &paths,
        TradingRule { profit_taking: 2.0, stop_loss: 0.5 },
        59,
        1.0,
    )?;
    assert!(wide.sharpe > tight.sharpe && wide.win_rate > 0.95);
    Ok(())
}
