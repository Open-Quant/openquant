//! The Rust examples on the cla and portfolio_optimization docs pages, run as tests.
//! `check:examples` only compiles page snippets; these execute them.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn cla_page() -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::DMatrix;
    use openquant::cla::{ClaError, WeightBounds, CLA};

    let mu = DMatrix::from_column_slice(4, 1, &[0.03, 0.07, 0.09, 0.04]);
    let vol = [0.05, 0.16, 0.22, 0.15];
    let rho =
        [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]];
    let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);

    let mut cla = CLA::new(WeightBounds::Tuple(0.0, 1.0), "mean");
    cla.allocate(None, Some(&mu), Some(&cov), None, None)?;

    // The walk starts at the highest-return asset and ends at minimum variance, where lambda is 0.
    assert_eq!(cla.weights.first().unwrap(), &vec![0.0, 0.0, 1.0, 0.0]);
    assert_eq!(*cla.lambdas.last().unwrap(), 0.0);
    // Every turning point is a feasible portfolio.
    for w in &cla.weights {
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(w.iter().all(|x| (-1e-9..=1.0 + 1e-9).contains(x)));
    }

    // With a 40% cap the frontier starts from the best feasible corner instead.
    let mut capped = CLA::new(WeightBounds::Tuple(0.0, 0.4), "mean");
    capped.allocate(None, Some(&mu), Some(&cov), None, Some("min_volatility"))?;
    assert!(capped.weights[0].iter().all(|x| *x <= 0.4 + 1e-9));

    assert!(matches!(
        cla.allocate(None, Some(&mu), Some(&cov), None, Some("max_return")),
        Err(ClaError::UnknownSolution(_))
    ));
    Ok(())
}

#[test]
fn portfolio_optimization_page() -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::DMatrix;
    use openquant::portfolio_optimization::{allocate_from_inputs, AllocError, AllocationOptions};

    let mu = [0.03, 0.07, 0.09, 0.04];
    let vol = [0.05, 0.16, 0.22, 0.15];
    let rho =
        [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]];
    let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);

    let min_vol = allocate_from_inputs(&mu, &cov, "min_volatility", &AllocationOptions::default())?;
    assert!((min_vol.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(min_vol.weights.iter().all(|w| *w > -1e-9));
    assert!((min_vol.portfolio_risk - 0.0476).abs() < 1e-4);

    // A return target above the minimum-variance return binds exactly, and costs risk.
    let target = AllocationOptions { target_return: 0.065, ..AllocationOptions::default() };
    let efficient = allocate_from_inputs(&mu, &cov, "efficient_risk", &target)?;
    assert!((efficient.portfolio_return - 0.065).abs() < 1e-7);
    assert!(efficient.portfolio_risk > min_vol.portfolio_risk);

    // With a 35% cap the most any portfolio can return is 6.8%, so 7% is infeasible...
    let capped = AllocationOptions {
        target_return: 0.07,
        tuple_bounds: Some((0.0, 0.35)),
        ..AllocationOptions::default()
    };
    assert!(matches!(
        allocate_from_inputs(&mu, &cov, "efficient_risk", &capped),
        Err(AllocError::OptimizationFailed(_))
    ));
    // ...and bounds that cannot sum to one are rejected before solving.
    let impossible =
        AllocationOptions { tuple_bounds: Some((0.3, 0.4)), ..AllocationOptions::default() };
    assert!(matches!(
        allocate_from_inputs(&mu, &cov, "min_volatility", &impossible),
        Err(AllocError::InfeasibleBounds { .. })
    ));
    Ok(())
}
