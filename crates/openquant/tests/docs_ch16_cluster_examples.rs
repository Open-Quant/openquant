//! The Rust examples on the hrp, hcaa and onc docs pages, run as tests. `check:examples` only
//! compiles page snippets; these execute them.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn hrp_page() -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::DMatrix;
    use openquant::hrp::{HierarchicalRiskParity, HrpError};

    // Assets 0 and 1 are nearly the same bet; asset 2 is independent. All have variance 0.04.
    let covariance = DMatrix::from_row_slice(
        3,
        3,
        &[0.040, 0.036, 0.000, 0.036, 0.040, 0.000, 0.000, 0.000, 0.040],
    );
    let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();

    let mut model = HierarchicalRiskParity::new();
    model.allocate(&names, None, None, Some(&covariance), None, false)?;

    // The tree joins a and b first, so they are adjacent in the leaf order.
    assert_eq!(model.clusters[0], [0, 1]);
    let position = |asset: usize| model.ordered_indices.iter().position(|&i| i == asset).unwrap();
    assert_eq!(position(0).abs_diff(position(1)), 1);

    assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(model.weights.iter().all(|w| *w > 0.0));
    // The independent asset gets more than either member of the correlated pair.
    assert!(model.weights[2] > model.weights[0] && model.weights[2] > model.weights[1]);

    assert_eq!(model.allocate(&names, None, None, None, None, false), Err(HrpError::NoData));
    Ok(())
}

#[test]
fn hcaa_page() -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::DMatrix;
    use openquant::hcaa::{HcaaError, HierarchicalClusteringAssetAllocation};

    let covariance = DMatrix::from_row_slice(
        3,
        3,
        &[0.010, 0.008, 0.000, 0.008, 0.010, 0.000, 0.000, 0.000, 0.040],
    );
    let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
    let mut model = HierarchicalClusteringAssetAllocation::new("mean");

    // Arguments: names, prices, returns, covariance, expected returns, metric, confidence level,
    // cluster count (ignored), resampling.
    model.allocate(
        &names,
        None,
        None,
        Some(&covariance),
        None,
        "minimum_variance",
        0.05,
        None,
        None,
    )?;
    let by_variance = model.weights.clone();
    model.allocate(
        &names,
        None,
        None,
        Some(&covariance),
        None,
        "minimum_standard_deviation",
        0.05,
        None,
        None,
    )?;

    // The volatile asset gets more under standard deviation than under variance.
    assert!(model.weights[2] > by_variance[2]);
    assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Tail metrics need returns, not just a covariance matrix.
    assert_eq!(
        model.allocate(
            &names,
            None,
            None,
            Some(&covariance),
            None,
            "expected_shortfall",
            0.05,
            None,
            None
        ),
        Err(HcaaError::MissingReturnsForTailRisk)
    );
    assert!(matches!(
        model.allocate(&names, None, None, Some(&covariance), None, "kelly", 0.05, None, None),
        Err(HcaaError::UnknownAllocationMetric(_))
    ));
    Ok(())
}

#[test]
fn onc_page() -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::DMatrix;
    use openquant::onc::{get_onc_clusters, OncError};

    // Two blocks of three: 0.8 within a block, 0.1 across.
    let block = |i: usize| i / 3;
    let corr = DMatrix::from_fn(6, 6, |i, j| {
        if i == j {
            1.0
        } else if block(i) == block(j) {
            0.8
        } else {
            0.1
        }
    });

    let result = get_onc_clusters(&corr, 3)?;
    assert_eq!(result.clusters.len(), 2);
    let mut found: Vec<Vec<usize>> = result.clusters.values().cloned().collect();
    found.sort();
    assert_eq!(found, vec![vec![0, 1, 2], vec![3, 4, 5]]);

    // One silhouette per item, all clearly positive for a clean partition.
    assert_eq!(result.silhouette_scores.len(), 6);
    assert!(result.silhouette_scores.iter().all(|s| *s > 0.5));

    assert_eq!(get_onc_clusters(&corr, 0).unwrap_err(), OncError::InvalidRepeat);
    Ok(())
}
