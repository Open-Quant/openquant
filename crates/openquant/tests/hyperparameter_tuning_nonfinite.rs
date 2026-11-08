use std::cell::Cell;
use std::collections::BTreeMap;

use chrono::{Duration, NaiveDate};
use openquant::cross_validation::SimpleClassifier;
use openquant::hyperparameter_tuning::{
    classification_score, grid_search, randomized_search, sample_log_uniform, sample_param_sets,
    HyperParamValue, ParamSet, RandomParamDistribution, SearchData, SearchScoring, TuningError,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

const SCORINGS: [SearchScoring; 3] =
    [SearchScoring::Accuracy, SearchScoring::BalancedAccuracy, SearchScoring::NegLogLoss];

#[test]
fn nonfinite_labels_are_rejected_even_when_their_weight_is_zero() {
    for label in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for weight in [0.0, 1.0] {
            for scoring in SCORINGS {
                assert_eq!(
                    classification_score(&[label, 1.0], &[0.1, 0.9], Some(&[weight, 1.0]), scoring),
                    Err(TuningError::NonBinaryLabels),
                    "label={label}, weight={weight}, scoring={scoring:?}"
                );
            }
        }
    }
}

#[test]
fn nonfinite_weights_are_rejected_for_every_score() {
    for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for scoring in SCORINGS {
            let result =
                classification_score(&[0.0, 1.0], &[0.1, 0.9], Some(&[weight, 1.0]), scoring);
            assert_eq!(
                result,
                Err(TuningError::NonFiniteSampleWeight),
                "weight={weight}, scoring={scoring:?}"
            );
        }
    }
}

#[test]
fn overflowing_finite_weight_sum_is_rejected_for_every_score() {
    for scoring in SCORINGS {
        let result =
            classification_score(&[0.0, 1.0], &[0.1, 0.9], Some(&[f64::MAX, f64::MAX]), scoring);
        assert_eq!(result, Err(TuningError::NonFiniteSampleWeightSum), "scoring={scoring:?}");
    }
}

#[test]
fn overflowing_finite_weight_loss_product_is_rejected() {
    let result =
        classification_score(&[1.0], &[0.01], Some(&[f64::MAX]), SearchScoring::NegLogLoss);
    assert_eq!(result, Err(TuningError::NonFiniteScore));
}

#[test]
fn overflowing_sum_of_finite_weighted_losses_is_rejected() {
    // Each loss term is finite, as is the sum of the weights, but the loss sum overflows.
    let weight = f64::MAX / 4.0;
    assert!((-weight * 0.1_f64.ln()).is_finite());
    assert!((weight + weight).is_finite());
    let result = classification_score(
        &[1.0, 1.0],
        &[0.1, 0.1],
        Some(&[weight, weight]),
        SearchScoring::NegLogLoss,
    );
    assert_eq!(result, Err(TuningError::NonFiniteScore));
}

#[test]
fn unused_log_loss_overflow_does_not_invalidate_accuracy() {
    for scoring in [SearchScoring::Accuracy, SearchScoring::BalancedAccuracy] {
        assert_eq!(classification_score(&[1.0], &[0.01], Some(&[f64::MAX]), scoring), Ok(0.0));
    }
}

#[test]
fn ordinary_weighted_scores_and_zero_weight_behavior_are_preserved() {
    let labels = [1.0, 1.0, 0.0, 0.0];
    let probabilities = [0.9, 0.2, 0.1, 1.0];
    // The last (confidently wrong) prediction has no weight.
    let weights = [1.0, 3.0, 2.0, 0.0];
    for (scoring, expected) in [
        (SearchScoring::Accuracy, 0.5),
        (SearchScoring::BalancedAccuracy, 0.625),
        (SearchScoring::NegLogLoss, (0.9_f64.ln() + 0.2_f64.ln()) / 2.0),
    ] {
        let actual =
            classification_score(&labels, &probabilities, Some(&weights), scoring).unwrap();
        assert!((actual - expected).abs() < 1e-14, "{scoring:?}: {actual}");
        assert_eq!(
            classification_score(&labels, &probabilities, Some(&[0.0; 4]), scoring),
            Err(TuningError::ZeroSampleWeightSum)
        );
        assert_eq!(
            classification_score(&labels, &probabilities, Some(&[-1.0, 1.0, 1.0, 1.0]), scoring),
            Err(TuningError::NegativeSampleWeight)
        );
    }
}

struct ConstantClassifier;

impl SimpleClassifier for ConstantClassifier {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        vec![0.5; x.len()]
    }
}

#[test]
fn search_rejects_invalid_data_before_building_any_classifier() {
    let x = vec![vec![0.0]; 6];
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let spans: Vec<_> = (0..6)
        .map(|i| (start + Duration::hours(i * 2), start + Duration::hours(i * 2 + 1)))
        .collect();
    let grid = BTreeMap::from([("p".to_owned(), vec![HyperParamValue::Float(0.5)])]);
    let space = BTreeMap::from([(
        "p".to_owned(),
        RandomParamDistribution::Choice(vec![HyperParamValue::Float(0.5)]),
    )]);
    for (label, weight) in [
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (f64::NEG_INFINITY, 1.0),
        (2.0, 1.0),
        (f64::from_bits(1), 0.0),
        (1.0_f64.next_down(), 1.0),
        (1e-12, 1.0),
        (1.0, f64::NAN),
        (1.0, f64::INFINITY),
        (1.0, f64::NEG_INFINITY),
        (1.0, -1.0),
    ] {
        let y = [1.0, 0.0, 1.0, 0.0, 1.0, label];
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0, weight];
        for randomized in [false, true] {
            let built = Cell::new(0);
            let build = |_: &ParamSet| {
                built.set(built.get() + 1);
                ConstantClassifier
            };
            let data = SearchData {
                x: &x,
                y: &y,
                sample_weight: Some(&weights),
                samples_info_sets: &spans,
            };
            let result = if randomized {
                randomized_search(build, &space, 1, 7, data, 3, 0.0, SearchScoring::Accuracy)
            } else {
                grid_search(build, &grid, data, 3, 0.0, SearchScoring::Accuracy)
            };
            assert!(result.is_err(), "label={label}, weight={weight}: {result:?}");
            assert_eq!(built.get(), 0, "invalid data reached the classifier");
        }
    }
}

#[test]
fn overflowing_search_weight_sum_is_rejected_before_building_a_classifier() {
    let x = vec![vec![0.0]; 6];
    let y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let weights = [f64::MAX; 6];
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let spans: Vec<_> = (0..6)
        .map(|i| (start + Duration::hours(i * 2), start + Duration::hours(i * 2 + 1)))
        .collect();
    let grid = BTreeMap::from([("p".to_owned(), vec![HyperParamValue::Float(0.5)])]);
    let space = BTreeMap::from([(
        "p".to_owned(),
        RandomParamDistribution::Choice(vec![HyperParamValue::Float(0.5)]),
    )]);
    for scoring in SCORINGS {
        for randomized in [false, true] {
            let built = Cell::new(0);
            let build = |_: &ParamSet| {
                built.set(built.get() + 1);
                ConstantClassifier
            };
            let data = SearchData {
                x: &x,
                y: &y,
                sample_weight: Some(&weights),
                samples_info_sets: &spans,
            };
            let result = if randomized {
                randomized_search(build, &space, 1, 7, data, 3, 0.0, scoring)
            } else {
                grid_search(build, &grid, data, 3, 0.0, scoring)
            };
            assert_eq!(
                result,
                Err(TuningError::NonFiniteSampleWeightSum),
                "randomized={randomized}, scoring={scoring:?}"
            );
            assert_eq!(built.get(), 0, "overflowing weights reached the classifier");
        }
    }
}

#[test]
fn finite_search_ties_still_select_the_last_candidate() {
    let x = vec![vec![0.0]; 6];
    let y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let spans: Vec<_> = (0..6)
        .map(|i| (start + Duration::hours(i * 2), start + Duration::hours(i * 2 + 1)))
        .collect();
    let grid =
        BTreeMap::from([("id".to_owned(), vec![HyperParamValue::Int(1), HyperParamValue::Int(2)])]);
    let result = grid_search(
        |_| ConstantClassifier,
        &grid,
        SearchData { x: &x, y: &y, sample_weight: None, samples_info_sets: &spans },
        3,
        0.0,
        SearchScoring::Accuracy,
    )
    .unwrap();
    assert_eq!(result.best_params["id"], HyperParamValue::Int(2));
    assert_eq!(result.best_score, 0.5);
    assert!(result.trials.iter().all(|trial| trial.fold_scores == [0.5; 3]));
}

#[test]
fn log_uniform_rejects_nonfinite_bounds_without_panicking() {
    for (low, high) in [
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (1.0, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, 1.0),
    ] {
        let mut rng = StdRng::seed_from_u64(7);
        let result = sample_log_uniform(low, high, &mut rng);
        assert_eq!(result, Err(TuningError::NonFiniteLogUniformBounds), "low={low}, high={high}");
    }
}

#[test]
fn random_parameter_sampling_rejects_nonfinite_log_uniform_bounds() {
    for high in [f64::NAN, f64::INFINITY] {
        let space = BTreeMap::from([(
            "p".to_owned(),
            RandomParamDistribution::LogUniform { low: 1.0, high },
        )]);
        assert_eq!(sample_param_sets(&space, 1, 7), Err(TuningError::NonFiniteLogUniformBounds));
    }
}

#[test]
fn log_uniform_adjacent_finite_bounds_return_the_only_in_range_value() {
    for low in [1e-300_f64, 1e300] {
        let high = low.next_up();
        assert!(low < high);
        assert_eq!(low.ln(), high.ln());
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(sample_log_uniform(low, high, &mut rng), Ok(low));
        let space =
            BTreeMap::from([("p".to_owned(), RandomParamDistribution::LogUniform { low, high })]);
        let draws = sample_param_sets(&space, 1, 7).unwrap();
        assert_eq!(draws[0]["p"], HyperParamValue::Float(low));
    }
}

#[test]
fn log_uniform_narrow_extreme_ranges_stay_inside_half_open_bounds() {
    for (low, high) in [(1e300, 1.000000000001e300), (1e-300, 1.000000000001e-300)] {
        let space =
            BTreeMap::from([("p".to_owned(), RandomParamDistribution::LogUniform { low, high })]);
        for draw in sample_param_sets(&space, 1000, 42).unwrap() {
            let value = draw["p"].as_f64().unwrap();
            assert!((low..high).contains(&value), "value={value:e}, bounds=[{low:e}, {high:e})");
        }
    }
}

#[test]
fn uniform_rejects_finite_bounds_whose_width_overflows_without_panicking() {
    for (low, high) in [(-1e308, 1e308), (-f64::MAX, f64::MAX)] {
        let space =
            BTreeMap::from([("p".to_owned(), RandomParamDistribution::Uniform { low, high })]);
        assert_eq!(sample_param_sets(&space, 3, 42), Err(TuningError::InvalidUniformBounds));
    }
}

#[test]
fn uniform_keeps_extreme_bounds_with_representable_width() {
    for (low, high) in [(-f64::MAX, 0.0), (0.0, f64::MAX), (-1.0, 1.0)] {
        let space =
            BTreeMap::from([("p".to_owned(), RandomParamDistribution::Uniform { low, high })]);
        for draw in sample_param_sets(&space, 100, 42).unwrap() {
            let value = draw["p"].as_f64().unwrap();
            assert!((low..high).contains(&value), "value={value:e}, bounds=[{low:e}, {high:e})");
        }
    }
}

#[test]
fn near_binary_labels_are_rejected_for_all_scores_and_weights() {
    for label in [
        f64::from_bits(1),
        -f64::from_bits(1),
        1e-12,
        -1e-12,
        1.0_f64.next_down(),
        1.0_f64.next_up(),
        1.0 - 5e-13,
        1.0 + 5e-13,
    ] {
        for weight in [0.0, 1.0] {
            for scoring in SCORINGS {
                assert_eq!(
                    classification_score(&[label, 1.0], &[0.1, 0.9], Some(&[weight, 1.0]), scoring),
                    Err(TuningError::NonBinaryLabels)
                );
            }
        }
    }
}

#[test]
fn exact_labels_including_signed_zero_remain_valid() {
    for zero in [0.0, -0.0] {
        for scoring in SCORINGS {
            let actual = classification_score(&[zero, 1.0], &[0.1, 0.9], None, scoring).unwrap();
            let expected =
                if matches!(scoring, SearchScoring::NegLogLoss) { 0.9_f64.ln() } else { 1.0 };
            assert!((actual - expected).abs() < 1e-15);
        }
    }
}
