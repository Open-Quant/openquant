//! Barriers must be finite positive widths, not disabled comparisons.
use openquant::synthetic_backtesting::{
    evaluate_rule_on_paths, search_optimal_trading_rule, OuProcessParams, StabilityCriteria,
    SyntheticBacktestError, TradingRule,
};

fn params() -> OuProcessParams {
    OuProcessParams {
        phi: 0.9,
        intercept: 0.0,
        equilibrium: 0.0,
        sigma: 1.0,
        r_squared: 0.0,
        stationary: true,
    }
}

#[test]
fn nonfinite_rule_barriers_are_rejected_by_name() {
    let paths = vec![vec![0.0, 1.0, -1.0]];
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for name in ["profit_taking", "stop_loss"] {
            let rule = TradingRule {
                profit_taking: if name == "profit_taking" { value } else { 1.0 },
                stop_loss: if name == "stop_loss" { value } else { 1.0 },
            };
            assert_eq!(
                evaluate_rule_on_paths(&paths, rule, 2, 1.0),
                Err(SyntheticBacktestError::Invalid { name, requirement: "finite and > 0" }),
                "{name} = {value}"
            );
        }
    }
}

#[test]
fn nonfinite_search_grid_barriers_are_rejected() {
    let paths = vec![vec![0.0, 1.0, -1.0]];
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for name in ["profit_taking", "stop_loss"] {
            let pt = if name == "profit_taking" { vec![1.0, value] } else { vec![1.0] };
            let sl = if name == "stop_loss" { vec![1.0, value] } else { vec![1.0] };
            assert_eq!(
                search_optimal_trading_rule(
                    params(),
                    &paths,
                    &pt,
                    &sl,
                    2,
                    1.0,
                    StabilityCriteria::default()
                ),
                Err(SyntheticBacktestError::Invalid { name, requirement: "finite and > 0" }),
                "{name} = {value}"
            );
        }
    }
}

#[test]
fn finite_nonpositive_barriers_keep_the_existing_error() {
    let paths = vec![vec![0.0, 1.0]];
    for value in [0.0, -0.0, -1.0] {
        for rule in [
            TradingRule { profit_taking: value, stop_loss: 1.0 },
            TradingRule { profit_taking: 1.0, stop_loss: value },
        ] {
            assert_eq!(
                evaluate_rule_on_paths(&paths, rule, 1, 1.0),
                Err(SyntheticBacktestError::NonPositiveBarriers)
            );
        }
    }
}

#[test]
fn smallest_positive_barriers_remain_valid() {
    let smallest = f64::from_bits(1);
    let paths = vec![vec![0.0, smallest], vec![0.0, -smallest]];
    let rule = TradingRule { profit_taking: smallest, stop_loss: smallest };
    let result = evaluate_rule_on_paths(&paths, rule, 1, 1.0).unwrap();
    assert_eq!(result.rule, rule);
    assert_eq!(result.mean_return, 0.0);
    assert_eq!(result.win_rate, 0.5);
    assert_eq!(result.avg_holding_steps, 1.0);
}
