//! Every public error is a real `std::error::Error`, and none is a bare `String` (issue #35).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Fails to compile if `T` is not an error that can cross threads and be boxed.
fn assert_error<T: std::error::Error + Send + Sync + 'static>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}

fn asserted() -> BTreeSet<&'static str> {
    BTreeSet::from([
        assert_error::<openquant::backtesting_engine::BacktestError>(),
        assert_error::<openquant::bet_sizing::BetSizingError>(),
        assert_error::<openquant::cla::ClaError>(),
        assert_error::<openquant::codependence::CodependenceError>(),
        assert_error::<openquant::combinatorial_optimization::CombinatorialOptimizationError>(),
        assert_error::<openquant::cross_validation::CrossValidationError>(),
        assert_error::<openquant::data_processing::DataProcessingError>(),
        assert_error::<openquant::dynamic_allocation::DynamicAllocationError>(),
        assert_error::<openquant::ensemble_methods::EnsembleError>(),
        assert_error::<openquant::etf_trick::EtfTrickError>(),
        assert_error::<openquant::feature_importance::FeatureImportanceError>(),
        assert_error::<openquant::filters::FilterError>(),
        assert_error::<openquant::fingerprint::FingerprintError>(),
        assert_error::<openquant::hcaa::HcaaError>(),
        assert_error::<openquant::hpc_parallel::HpcParallelError>(),
        assert_error::<openquant::hrp::HrpError>(),
        assert_error::<openquant::hyperparameter_tuning::TuningError>(),
        assert_error::<openquant::microstructural_features::MicrostructuralError>(),
        assert_error::<openquant::onc::OncError>(),
        assert_error::<openquant::pipeline::PipelineError>(),
        assert_error::<openquant::portfolio_optimization::AllocError>(),
        assert_error::<openquant::risk_metrics::RiskMetricsError>(),
        assert_error::<openquant::sample_weights::SampleWeightsError>(),
        assert_error::<openquant::sb_bagging::SbBaggingError>(),
        assert_error::<openquant::strategy_risk::StrategyRiskError>(),
        assert_error::<openquant::streaming_hpc::StreamingHpcError>(),
        assert_error::<openquant::structural_breaks::StructuralBreakError>(),
        assert_error::<openquant::synthetic_backtesting::SyntheticBacktestError>(),
        assert_error::<openquant::util::InputError>(),
    ])
}

fn source_files(dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(fs::read_to_string(&path).unwrap());
        }
    }
}

fn sources() -> Vec<String> {
    let mut out = Vec::new();
    source_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut out);
    out
}

/// The list above is checked by the compiler; this checks that the list is complete, so a new
/// `pub enum FooError` without an `Error` impl cannot slip past it.
#[test]
fn every_public_error_enum_is_asserted() {
    let declared: BTreeSet<String> = sources()
        .iter()
        .flat_map(|src| src.lines())
        .filter_map(|line| line.strip_prefix("pub enum "))
        .map(|rest| rest.split(|c: char| !c.is_alphanumeric()).next().unwrap().to_string())
        .filter(|name| name.ends_with("Error"))
        .collect();
    let asserted: BTreeSet<String> = asserted().into_iter().map(String::from).collect();
    assert_eq!(declared, asserted);
}

#[test]
fn no_public_function_returns_a_string_error() {
    let offenders: Vec<String> = sources()
        .iter()
        .flat_map(|src| src.lines())
        .filter(|line| line.contains(", String>") && line.contains("Result<"))
        .map(|line| line.trim().to_string())
        .collect();
    assert!(offenders.is_empty(), "{offenders:#?}");
}

/// The typed errors replaced hand-written strings; the text callers see must not have moved.
#[test]
fn messages_are_unchanged_by_the_move_to_typed_errors() {
    use openquant::backtesting_engine::BacktestError;
    use openquant::cross_validation::{CrossValidationError, PurgedKFold};
    use openquant::ensemble_methods::EnsembleError;
    use openquant::hyperparameter_tuning::TuningError;
    use openquant::microstructural_features::{encode_tick_rule_array, quantile_mapping};

    assert_eq!(BacktestError::Empty("returns").to_string(), "returns cannot be empty");
    assert_eq!(
        BacktestError::Invalid { name: "min_train_size", requirement: "> 0" }.to_string(),
        "min_train_size must be > 0"
    );
    assert_eq!(
        BacktestError::GroupOccurrences { group: 2, found: 1, expected: 3 }.to_string(),
        "group 2 has 1 occurrences, expected 3"
    );
    assert_eq!(
        EnsembleError::NonBinaryLabels.to_string(),
        "classification vote expects binary labels in {0,1}"
    );
    assert_eq!(
        TuningError::EmptyGridEntry("depth".into()).to_string(),
        "param_grid entry 'depth' cannot be empty"
    );
    let day =
        |d| chrono::NaiveDate::from_ymd_opt(2024, 1, d).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let spans = vec![(day(1), day(2)), (day(2), day(3))];
    // PurgedKFold is not Debug, so unwrap_err is unavailable.
    let err = PurgedKFold::new(1, spans, 0.0).err().expect("one split is invalid");
    assert_eq!(err, CrossValidationError::InvalidSplits { n_splits: 1, n_samples: 2 });
    assert_eq!(err.to_string(), "n_splits must be between 2 and the number of samples (2), got 1");
    assert_eq!(
        encode_tick_rule_array(&[1, 7]).unwrap_err().to_string(),
        "Unknown value for tick rule: 7"
    );
    assert_eq!(quantile_mapping(&[], 2).unwrap_err().to_string(), "array must not be empty");
}

/// A wrapped error reports the inner message and keeps the inner value reachable.
#[test]
fn cross_module_errors_convert_with_question_mark() {
    use openquant::cross_validation::CrossValidationError;
    use openquant::hyperparameter_tuning::TuningError;

    let inner = CrossValidationError::Empty("samples_info_sets");
    let outer: TuningError = inner.clone().into();
    assert_eq!(outer.to_string(), inner.to_string());
    assert_eq!(outer, TuningError::CrossValidation(inner));
}
