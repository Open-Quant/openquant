use openquant::synthetic_backtesting::{
    calibrate_ou_params, detect_no_stable_optimum, generate_ou_paths, run_synthetic_otr_workflow,
    search_optimal_trading_rule, OuProcessParams, RuleSurfacePoint, StabilityCriteria,
    SyntheticBacktestConfig, TradingRule,
};

#[test]
fn test_generate_ou_paths_is_seeded_and_reproducible() {
    let params = OuProcessParams {
        phi: 0.85,
        intercept: 15.0,
        equilibrium: 100.0,
        sigma: 1.25,
        r_squared: 0.9,
        stationary: true,
    };

    let p1 = generate_ou_paths(params, 98.0, 16, 64, 42).unwrap();
    let p2 = generate_ou_paths(params, 98.0, 16, 64, 42).unwrap();
    let p3 = generate_ou_paths(params, 98.0, 16, 64, 43).unwrap();

    assert_eq!(p1, p2);
    assert_ne!(p1, p3);
}

#[test]
fn test_calibration_recovers_ou_phi_reasonably() {
    let true_params = OuProcessParams {
        phi: 0.82,
        intercept: 18.0,
        equilibrium: 100.0,
        sigma: 0.8,
        r_squared: 0.0,
        stationary: true,
    };
    let paths = generate_ou_paths(true_params, 100.0, 1, 1200, 7).unwrap();
    let fit = calibrate_ou_params(&paths[0]).unwrap();

    assert!((fit.phi - true_params.phi).abs() < 0.08);
    assert!(fit.sigma > 0.0);
    assert!(fit.stationary);
}

#[test]
fn test_regime_contrast_mean_reverting_vs_random_walk_like() {
    let mean_reverting = OuProcessParams {
        phi: 0.65,
        intercept: 0.35,
        equilibrium: 1.0,
        sigma: 1.0,
        r_squared: 0.0,
        stationary: true,
    };
    let random_walk_like = OuProcessParams {
        phi: 0.995,
        intercept: 0.0,
        equilibrium: 0.0,
        sigma: 1.0,
        r_squared: 0.0,
        stationary: true,
    };

    let pt_grid = vec![0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
    let sl_grid = vec![0.5, 1.0, 2.0, 3.0, 4.0];
    let criteria = StabilityCriteria::default();

    let mr_paths = generate_ou_paths(mean_reverting, 0.0, 4_000, 128, 11).unwrap();
    let rw_paths = generate_ou_paths(random_walk_like, 0.0, 4_000, 128, 11).unwrap();

    let mr = search_optimal_trading_rule(
        mean_reverting,
        &mr_paths,
        &pt_grid,
        &sl_grid,
        64,
        1.0,
        criteria,
    )
    .unwrap();
    let rw = search_optimal_trading_rule(
        random_walk_like,
        &rw_paths,
        &pt_grid,
        &sl_grid,
        64,
        1.0,
        criteria,
    )
    .unwrap();

    assert!(mr.best_point.sharpe > rw.best_point.sharpe + 0.1);
    assert!(!mr.diagnostics.no_stable_optimum);
    assert!(rw.diagnostics.no_stable_optimum);
}

#[test]
fn test_detect_no_stable_optimum_for_flat_surface() {
    let flat = vec![
        RuleSurfacePoint {
            rule: TradingRule { profit_taking: 1.0, stop_loss: 1.0 },
            sharpe: 0.04,
            mean_return: 0.01,
            std_return: 0.25,
            win_rate: 0.50,
            avg_holding_steps: 10.0,
        },
        RuleSurfacePoint {
            rule: TradingRule { profit_taking: 1.0, stop_loss: 2.0 },
            sharpe: 0.05,
            mean_return: 0.01,
            std_return: 0.25,
            win_rate: 0.50,
            avg_holding_steps: 10.0,
        },
        RuleSurfacePoint {
            rule: TradingRule { profit_taking: 2.0, stop_loss: 1.0 },
            sharpe: 0.03,
            mean_return: 0.01,
            std_return: 0.25,
            win_rate: 0.50,
            avg_holding_steps: 10.0,
        },
    ];

    let diag = detect_no_stable_optimum(&flat, 0.99, StabilityCriteria::default()).unwrap();
    assert!(diag.no_stable_optimum);
    assert!(diag.reason.contains("no stable optimum"));
}

#[test]
fn test_run_synthetic_otr_workflow_end_to_end() {
    let params = OuProcessParams {
        phi: 0.75,
        intercept: 0.5,
        equilibrium: 2.0,
        sigma: 1.0,
        r_squared: 0.0,
        stationary: true,
    };
    let historical = generate_ou_paths(params, 0.0, 1, 800, 9).unwrap();
    let config = SyntheticBacktestConfig {
        initial_price: 0.0,
        n_paths: 1500,
        horizon: 96,
        seed: 77,
        profit_taking_grid: vec![0.5, 1.0, 2.0, 3.0],
        stop_loss_grid: vec![0.5, 1.0, 2.0, 3.0],
        max_holding_steps: 64,
        annualization_factor: 1.0,
        stability_criteria: StabilityCriteria::default(),
    };

    let result = run_synthetic_otr_workflow(&historical[0], &config).unwrap();
    assert_eq!(result.response_surface.len(), 16);
    assert!(result.best_point.sharpe.is_finite());
    assert!(result.params.sigma > 0.0);
}
