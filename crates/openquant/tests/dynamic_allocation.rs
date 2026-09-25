//! AFML Chapter 21 dynamic allocation: partitions, Ω and the trajectory search (issue #112).

use nalgebra::DMatrix;
use openquant::dynamic_allocation::{
    all_weights, dynamic_optimal_portfolio, partition_count, pigeonhole_partitions,
    trajectory_sharpe_ratio, transaction_costs, weight_count, DynamicAllocationConfig,
    DynamicAllocationError, HorizonForecast,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12, "{a} != {b}");
}

// --- Snippet 21.1 ----------------------------------------------------------------------------

#[test]
fn partitions_match_stars_and_bars_and_are_valid() {
    for n in 1..=5usize {
        for k in 0..=6usize {
            let parts = pigeonhole_partitions(k, n);
            let expected = binomial((k + n - 1) as u64, (n - 1) as u64);
            assert_eq!(parts.len() as u64, expected, "k={k} n={n}");
            assert_eq!(partition_count(k, n), Some(expected as u128), "k={k} n={n}");
            for p in &parts {
                assert_eq!(p.len(), n);
                assert_eq!(p.iter().sum::<usize>(), k);
            }
            let mut unique = parts.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), parts.len(), "duplicate partition for k={k} n={n}");
        }
    }
}

#[test]
fn partitions_follow_the_books_order() {
    // Python: [r for r in pigeonHole(2, 3)], from combinations_with_replacement(range(3), 2).
    assert_eq!(
        pigeonhole_partitions(2, 3),
        vec![
            vec![2, 0, 0],
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![0, 2, 0],
            vec![0, 1, 1],
            vec![0, 0, 2],
        ]
    );
}

#[test]
fn partition_edge_cases_match_python() {
    // combinations_with_replacement(range(0), 0) yields one empty tuple, and nothing for k > 0.
    assert_eq!(pigeonhole_partitions(0, 0), vec![Vec::<usize>::new()]);
    assert!(pigeonhole_partitions(3, 0).is_empty());
    assert_eq!(pigeonhole_partitions(0, 3), vec![vec![0, 0, 0]]);
    assert_eq!(partition_count(3, 0), Some(0));
    assert_eq!(partition_count(0, 0), Some(1));
    assert_eq!(partition_count(usize::MAX, usize::MAX), None);
}

// --- Snippet 21.2 ----------------------------------------------------------------------------

/// Snippet 21.2 transcribed literally: every partition times all 2^n sign patterns, duplicates
/// included.
fn book_all_weights(k: usize, n: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for part in pigeonhole_partitions(k, n) {
        for pattern in 0..(1usize << n) {
            out.push(
                (0..n)
                    .map(|i| {
                        let sign = if (pattern >> (n - 1 - i)) & 1 == 1 { 1.0 } else { -1.0 };
                        sign * part[i] as f64 / k as f64
                    })
                    .collect(),
            );
        }
    }
    out
}

fn first_occurrences(list: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for w in list {
        // -0.0 == 0.0, so a zero with either sign is the same vector.
        if !out.iter().any(|seen| seen == w) {
            out.push(w.clone());
        }
    }
    out
}

#[test]
fn sign_pattern_counts() {
    for n in 1..=4usize {
        for k in 1..=5usize {
            let omega = all_weights(k, n).unwrap();
            let book = book_all_weights(k, n);
            // The book's list: 2^n sign patterns of each of C(k+n-1, n-1) partitions.
            assert_eq!(
                book.len() as u64,
                (1u64 << n) * binomial((k + n - 1) as u64, (n - 1) as u64)
            );
            // Distinct vectors: choose j non-zero assets, split k units among them, sign them.
            let distinct: u64 = (1..=k.min(n) as u64)
                .map(|j| binomial(n as u64, j) * binomial(k as u64 - 1, j - 1) * (1 << j))
                .sum();
            assert_eq!(omega.len() as u64, distinct, "k={k} n={n}");
            assert_eq!(weight_count(k, n), Some(distinct as u128));
            // Same set, same order of first occurrence, as the book's list.
            assert_eq!(omega, first_occurrences(&book), "k={k} n={n}");
            for w in &omega {
                assert_close(w.iter().map(|x| x.abs()).sum(), 1.0);
            }
        }
    }
}

#[test]
fn omega_for_one_unit_in_two_assets() {
    assert_eq!(
        all_weights(1, 2).unwrap(),
        vec![vec![-1.0, 0.0], vec![1.0, 0.0], vec![0.0, -1.0], vec![0.0, 1.0]]
    );
    assert_eq!(all_weights(3, 3).unwrap().len(), 38);
    assert_eq!(all_weights(0, 2), Err(DynamicAllocationError::ZeroCount("the number of units k")));
    assert_eq!(all_weights(2, 0), Err(DynamicAllocationError::ZeroCount("the number of assets n")));
}

// --- Snippet 21.3 ----------------------------------------------------------------------------

fn diag(values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(values))
}

/// Two assets, two horizons: asset 0 does well then badly, asset 1 does well throughout and is
/// less volatile.
fn two_asset_forecasts(cost: f64) -> Vec<HorizonForecast> {
    vec![
        HorizonForecast {
            mean: vec![0.02, 0.01],
            covariance: diag(&[0.04, 0.01]),
            cost: vec![cost, cost],
        },
        HorizonForecast {
            mean: vec![-0.01, 0.03],
            covariance: diag(&[0.04, 0.01]),
            cost: vec![cost, cost],
        },
    ]
}

#[test]
fn transaction_costs_are_square_root_in_the_trade() {
    let horizons = vec![
        HorizonForecast {
            mean: vec![0.0, 0.0],
            covariance: diag(&[1.0, 1.0]),
            cost: vec![0.01, 0.02],
        },
        HorizonForecast {
            mean: vec![0.0, 0.0],
            covariance: diag(&[1.0, 1.0]),
            cost: vec![0.03, 0.04],
        },
    ];
    let trajectory = vec![vec![0.5, -0.5], vec![0.5, 0.5]];
    let costs = transaction_costs(&trajectory, &horizons, &[0.0, 0.0]).unwrap();
    // h=0: from cash, trade 0.5 in each: 0.01·√0.5 + 0.02·√0.5.
    assert_close(costs[0], 0.03 * 0.5_f64.sqrt());
    // h=1: asset 0 unchanged, asset 1 moves by 1: 0.03·0 + 0.04·√1.
    assert_close(costs[1], 0.04);
    // Starting from the first portfolio, the first horizon costs nothing.
    let costs = transaction_costs(&trajectory, &horizons, &[0.5, -0.5]).unwrap();
    assert_close(costs[0], 0.0);
}

#[test]
fn hand_worked_trajectory() {
    let horizons = two_asset_forecasts(0.001);
    // Switch from asset 0 to asset 1: mean 0.02 + 0.03, costs 0.001 then 0.001 + 0.001,
    // variance 0.04 + 0.01.
    let switch = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_eq!(transaction_costs(&switch, &horizons, &[0.0, 0.0]).unwrap(), vec![0.001, 0.002]);
    let sr = trajectory_sharpe_ratio(&switch, &horizons, &[0.0, 0.0]).unwrap();
    assert_close(sr, (0.05 - 0.003) / 0.05_f64.sqrt());

    // Holding asset 1 throughout: mean 0.01 + 0.03, one cost of 0.001, variance 0.01 + 0.01.
    // Of the 16 trajectories this is the best.
    let best = dynamic_optimal_portfolio(&horizons, &DynamicAllocationConfig::new(1)).unwrap();
    assert_eq!(best.weights, vec![vec![0.0, 1.0], vec![0.0, 1.0]]);
    assert_close(best.sharpe_ratio, (0.04 - 0.001) / 0.02_f64.sqrt());
    assert_eq!(best.transaction_costs, vec![0.001, 0.0]);
    assert_eq!(best.trajectories_evaluated, 16);
}

#[test]
fn costs_can_keep_the_initial_position() {
    // Starting fully in asset 0 with expensive trading, every trade costs more than it earns.
    let horizons = two_asset_forecasts(0.05);
    let config = DynamicAllocationConfig {
        initial_weights: Some(vec![1.0, 0.0]),
        ..DynamicAllocationConfig::new(1)
    };
    let best = dynamic_optimal_portfolio(&horizons, &config).unwrap();
    assert_eq!(best.weights, vec![vec![1.0, 0.0], vec![1.0, 0.0]]);
    assert_close(best.sharpe_ratio, 0.01 / 0.08_f64.sqrt());
    assert_eq!(best.transaction_costs, vec![0.0, 0.0]);

    // From cash, with free trading, it moves into asset 1 as in the hand-worked case.
    let best =
        dynamic_optimal_portfolio(&two_asset_forecasts(0.0), &DynamicAllocationConfig::new(1))
            .unwrap();
    assert_eq!(best.weights, vec![vec![0.0, 1.0], vec![0.0, 1.0]]);
}

// --- Brute force ---------------------------------------------------------------------------

fn random_forecasts(rng: &mut StdRng, n: usize, h: usize) -> Vec<HorizonForecast> {
    (0..h)
        .map(|_| {
            let a = DMatrix::from_fn(n, n, |_, _| rng.gen_range(-0.2..0.2));
            HorizonForecast {
                mean: (0..n).map(|_| rng.gen_range(-0.05..0.05)).collect(),
                covariance: &a * a.transpose() + DMatrix::identity(n, n) * 0.01,
                cost: (0..n).map(|_| rng.gen_range(0.0..0.01)).collect(),
            }
        })
        .collect()
}

/// `dynOptPort` transcribed literally: the book's Ω (duplicates included), the full Cartesian
/// product in `itertools.product` order, `evalTCosts` and `evalSR` written out, strict `<`.
fn book_dyn_opt_port(
    horizons: &[HorizonForecast],
    k: usize,
    initial: &[f64],
) -> (Vec<Vec<f64>>, f64) {
    let n = horizons[0].mean.len();
    let omega = book_all_weights(k, n);
    let h = horizons.len();
    let mut best: Option<(f64, Vec<Vec<f64>>)> = None;
    let total = omega.len().pow(h as u32);
    for mut code in 0..total {
        let mut idx = vec![0; h];
        for slot in (0..h).rev() {
            idx[slot] = code % omega.len();
            code /= omega.len();
        }
        let w: Vec<Vec<f64>> = idx.iter().map(|&i| omega[i].clone()).collect();
        let mut prev = initial.to_vec();
        let (mut mean, mut cov) = (0.0, 0.0);
        for (wh, f) in w.iter().zip(horizons) {
            let tcost: f64 = (0..n).map(|a| f.cost[a] * (wh[a] - prev[a]).abs().sqrt()).sum();
            mean += (0..n).map(|a| wh[a] * f.mean[a]).sum::<f64>() - tcost;
            for a in 0..n {
                for b in 0..n {
                    cov += wh[a] * f.covariance[(a, b)] * wh[b];
                }
            }
            prev = wh.clone();
        }
        let sr = mean / cov.sqrt();
        if best.as_ref().is_none_or(|(s, _)| *s < sr) {
            best = Some((sr, w));
        }
    }
    let (sr, w) = best.unwrap();
    (w, sr)
}

#[test]
fn search_matches_a_literal_brute_force() {
    let mut rng = StdRng::seed_from_u64(21);
    for (n, k, h) in [(2, 1, 3), (2, 3, 2), (3, 2, 2), (3, 3, 1), (2, 2, 3)] {
        for trial in 0..4 {
            let horizons = random_forecasts(&mut rng, n, h);
            let initial: Vec<f64> = if trial % 2 == 0 {
                vec![0.0; n]
            } else {
                all_weights(k, n).unwrap()[trial].clone()
            };
            let (book_w, book_sr) = book_dyn_opt_port(&horizons, k, &initial);
            let config = DynamicAllocationConfig {
                initial_weights: Some(initial.clone()),
                ..DynamicAllocationConfig::new(k)
            };
            let ours = dynamic_optimal_portfolio(&horizons, &config).unwrap();
            assert!((ours.sharpe_ratio - book_sr).abs() < 1e-12, "n={n} k={k} h={h}");
            assert_eq!(ours.weights, book_w, "n={n} k={k} h={h} trial={trial}");
            assert_close(
                trajectory_sharpe_ratio(&ours.weights, &horizons, &initial).unwrap(),
                ours.sharpe_ratio,
            );
            assert_eq!(ours.trajectories_evaluated, all_weights(k, n).unwrap().len().pow(h as u32));
        }
    }
}

// --- Guards ----------------------------------------------------------------------------------

#[test]
fn the_trajectory_cap_is_a_typed_error() {
    let mut rng = StdRng::seed_from_u64(1);
    let horizons = random_forecasts(&mut rng, 3, 5);
    // |Ω| = 38 for k = n = 3, so 38^5 = 79,235,168 trajectories.
    let err = dynamic_optimal_portfolio(&horizons, &DynamicAllocationConfig::new(3)).unwrap_err();
    assert_eq!(
        err,
        DynamicAllocationError::TooManyTrajectories {
            trajectories: 79_235_168,
            weights: 38,
            horizons: 5,
            max_trajectories: 1_000_000,
        }
    );
    assert!(err.to_string().contains("79235168 trajectories"));

    // Exactly at the cap is allowed.
    let small = &horizons[..2];
    let config =
        DynamicAllocationConfig { max_trajectories: 38 * 38, ..DynamicAllocationConfig::new(3) };
    assert_eq!(dynamic_optimal_portfolio(small, &config).unwrap().trajectories_evaluated, 1444);
    let config = DynamicAllocationConfig { max_trajectories: 38 * 38 - 1, ..config };
    assert!(matches!(
        dynamic_optimal_portfolio(small, &config),
        Err(DynamicAllocationError::TooManyTrajectories { .. })
    ));

    // A count that overflows saturates rather than panicking.
    let err =
        dynamic_optimal_portfolio(&horizons, &DynamicAllocationConfig::new(1_000_000)).unwrap_err();
    assert!(matches!(
        err,
        DynamicAllocationError::TooManyTrajectories { trajectories: u128::MAX, .. }
    ));
}

#[test]
fn invalid_inputs_are_rejected() {
    let good = two_asset_forecasts(0.001);
    let config = DynamicAllocationConfig::new(1);

    assert_eq!(
        dynamic_optimal_portfolio(&[], &config),
        Err(DynamicAllocationError::ZeroCount("the number of horizons"))
    );
    assert_eq!(
        dynamic_optimal_portfolio(&good, &DynamicAllocationConfig::new(0)),
        Err(DynamicAllocationError::ZeroCount("the number of units k"))
    );

    let mut bad = good.clone();
    bad[1].mean = vec![0.0; 3];
    assert!(matches!(
        dynamic_optimal_portfolio(&bad, &config),
        Err(DynamicAllocationError::DimensionMismatch { expected: 2, got: 3, .. })
    ));

    let mut bad = good.clone();
    bad[0].cost[1] = -0.001;
    assert_eq!(
        dynamic_optimal_portfolio(&bad, &config),
        Err(DynamicAllocationError::NegativeCost { horizon: 0 })
    );

    let mut bad = good.clone();
    bad[1].mean[0] = f64::NAN;
    assert!(matches!(
        dynamic_optimal_portfolio(&bad, &config),
        Err(DynamicAllocationError::NonFinite(_))
    ));

    let mut bad = good.clone();
    bad[1].covariance = DMatrix::from_row_slice(2, 2, &[0.01, 0.01, 0.01, 0.01]); // singular
    assert_eq!(
        dynamic_optimal_portfolio(&bad, &config),
        Err(DynamicAllocationError::CovarianceNotPositiveDefinite { horizon: 1 })
    );

    let mut bad = good.clone();
    bad[0].covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.01, 0.01]); // asymmetric
    assert_eq!(
        dynamic_optimal_portfolio(&bad, &config),
        Err(DynamicAllocationError::CovarianceNotPositiveDefinite { horizon: 0 })
    );

    let wrong_initial = DynamicAllocationConfig {
        initial_weights: Some(vec![1.0]),
        ..DynamicAllocationConfig::new(1)
    };
    assert!(matches!(
        dynamic_optimal_portfolio(&good, &wrong_initial),
        Err(DynamicAllocationError::DimensionMismatch { expected: 2, got: 1, .. })
    ));

    assert!(matches!(
        transaction_costs(&[vec![1.0, 0.0]], &good, &[0.0, 0.0]),
        Err(DynamicAllocationError::DimensionMismatch { expected: 2, got: 1, .. })
    ));
}
