//! Invalid arguments must come back as `Err`, never as a panic (issue #36).
//!
//! Each test feeds a public function the input that used to abort the process: an empty or
//! mismatched series, a zero window, a NaN, a non-ASCII message.

use chrono::{Duration, NaiveDate, NaiveDateTime};
use openquant::backtest_statistics::{deflated_sharpe_ratio, minimum_track_record_length};
use openquant::bet_sizing::{
    bet_size, bet_size_budget, bet_size_dynamic, bet_size_power, bet_size_reserve_full,
    get_concurrent_sides, get_target_pos_power, get_w_power, BetSizingError,
};
use openquant::data_structures::{
    imbalance_bars, run_bars, standard_bars, time_bars, ImbalanceBarType, StandardBarType, Trade,
};
use openquant::ef3m::{centered_moment, M2N};
use openquant::filters::{cusum_filter_indices, cusum_filter_timestamps, FilterError, Threshold};
use openquant::microstructural_features::{
    encode_array, get_bar_based_amihud_lambda, get_bar_based_hasbrouck_lambda,
    get_bar_based_kyle_lambda, get_bekker_parkinson_vol, get_bvc_buy_volume,
    get_corwin_schultz_estimator, get_konto_entropy, get_lempel_ziv_entropy, get_plug_in_entropy,
    get_roll_impact, get_shannon_entropy, get_trades_based_kyle_lambda, get_vpin, quantile_mapping,
    vwap,
};
use openquant::sampling::{
    bootstrap_loop_run, get_ind_mat_average_uniqueness, get_ind_mat_label_uniqueness,
    get_ind_matrix, num_concurrent_events, seq_bootstrap,
};
use openquant::util::fast_ewma::ewma;
use openquant::util::volatility::{get_garman_class_vol, get_parkinson_vol, get_yang_zhang_vol};
use openquant::util::InputError;

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, day).unwrap().and_hms_opt(0, 0, 0).unwrap()
}

fn trades() -> Vec<Trade> {
    (0..4)
        .map(|i| Trade {
            timestamp: ts(1) + Duration::seconds(i),
            price: 100.0 + i as f64,
            volume: 1.0,
        })
        .collect()
}

const LONG: [f64; 3] = [1.0, 2.0, 3.0];
const SHORT: [f64; 1] = [1.0];

#[test]
fn ewma_rejects_a_zero_window() {
    assert!(matches!(ewma(&LONG, 0), Err(InputError::OutOfRange { name: "window", .. })));
    // ... including for an empty series, which used to slip past the check.
    assert!(ewma(&[], 0).is_err());
    assert_eq!(ewma(&[], 3), Ok(vec![]));
}

#[test]
fn range_estimators_reject_mismatched_lengths() {
    let expected = Err(InputError::LengthMismatch { name: "low", len: 1, expected: 3 });
    assert_eq!(get_parkinson_vol(&LONG, &SHORT, 2), expected);
    assert_eq!(get_garman_class_vol(&LONG, &LONG, &SHORT, &LONG, 2), expected);
    assert_eq!(get_yang_zhang_vol(&LONG, &LONG, &SHORT, &LONG, 2), expected);
    assert!(get_garman_class_vol(&LONG, &LONG, &LONG, &SHORT, 2).is_err());
}

#[test]
fn deflated_sharpe_ratio_needs_two_estimates() {
    for estimates_param in [false, true] {
        for estimates in [&[][..], &[0.5][..]] {
            let got = deflated_sharpe_ratio(1.14, estimates, 250, 0.0, 3.0, estimates_param, false);
            assert!(matches!(got, Err(InputError::TooShort { min: 2, .. })), "{got:?}");
        }
    }
    // (std, number of trials) with a single trial: the expected maximum is undefined.
    assert!(deflated_sharpe_ratio(1.14, &[0.5, 1.0], 250, 0.0, 3.0, true, false).is_err());
    assert!(deflated_sharpe_ratio(1.14, &[0.5, 100.0], 250, 0.0, 3.0, true, false).is_ok());
}

#[test]
fn minimum_track_record_length_rejects_alpha_outside_the_unit_interval() {
    for alpha in [-0.1, 1.5, f64::NAN] {
        assert!(minimum_track_record_length(1.14, 1.0, 0.0, 3.0, alpha).is_err(), "{alpha}");
    }
    assert!(minimum_track_record_length(1.14, 1.0, 0.0, 3.0, 0.05).is_ok());
}

#[test]
fn bet_sizing_reports_bad_arguments() {
    assert_eq!(
        bet_size_power(2.0, 1.5),
        Err(BetSizingError::PriceDivergenceOutOfRange { value: 1.5 })
    );
    assert!(get_w_power(1.2, 0.8).is_err());
    assert!(get_target_pos_power(2.0, 3.0, 1.0, 10.0).is_err());
    assert!(matches!(bet_size(2.0, 0.4, "nope"), Err(BetSizingError::InvalidFunction { .. })));
    assert!(bet_size_dynamic(&[], &[], &[], &[]).is_err());
    assert!(bet_size_dynamic(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0], &[1.0]).is_err());
}

#[test]
fn concurrent_sides_reject_a_side_series_of_the_wrong_length() {
    let t1 = vec![(ts(1), ts(3)), (ts(2), ts(4))];
    let expected = BetSizingError::LengthMismatch { name: "side", len: 1, expected: 2 };
    assert_eq!(get_concurrent_sides(&t1, &[1.0]), Err(expected.clone()));
    assert_eq!(bet_size_budget(&t1, &[1.0]), Err(expected));
    assert_eq!(
        bet_size_reserve_full(&[], &[], 1, 1e-5, 10, false),
        Err(BetSizingError::EmptyInput("t1"))
    );
}

#[test]
fn filters_report_short_thresholds_and_timestamps() {
    let close = [100.0, 101.0, 99.0, 102.0];
    assert!(matches!(
        cusum_filter_indices(&close, Threshold::Dynamic(vec![0.001])),
        Err(FilterError::MissingDynamicThreshold { .. })
    ));
    assert!(matches!(
        cusum_filter_timestamps(&close, &[ts(1)], Threshold::Scalar(0.001)),
        Err(FilterError::TimestampIndexOutOfBounds { .. })
    ));
}

#[test]
fn bar_builders_reject_non_positive_thresholds() {
    let trades = trades();
    for bad in [0.0, -1.0, f64::NAN] {
        assert!(standard_bars(&trades, bad, StandardBarType::Tick).is_err(), "{bad}");
        assert!(imbalance_bars(&trades, bad, ImbalanceBarType::Tick).is_err(), "{bad}");
    }
    assert!(run_bars(&trades, 0).is_err());
    assert!(time_bars(&trades, Duration::zero()).is_err());
    assert!(time_bars(&trades, Duration::seconds(-5)).is_err());
    // Empty input is valid and yields no bars.
    assert_eq!(standard_bars(&[], 2.0, StandardBarType::Tick).unwrap().len(), 0);
    assert_eq!(standard_bars(&trades, 2.0, StandardBarType::Tick).unwrap().len(), 2);
}

#[test]
fn indicator_matrix_functions_reject_malformed_input() {
    assert!(get_ind_matrix(&[(5, 2)], &[0, 1, 2]).is_err());

    let ragged = vec![vec![1, 0], vec![1]];
    assert!(get_ind_mat_average_uniqueness(&ragged).is_err());
    assert!(get_ind_mat_label_uniqueness(&ragged).is_err());
    assert!(bootstrap_loop_run(&ragged, &[0.0, 0.0]).is_err());
    assert!(seq_bootstrap(&ragged, None, None).is_err());

    let ind = vec![vec![1, 0], vec![1, 1]];
    assert!(bootstrap_loop_run(&ind, &[0.0]).is_err());
    assert!(seq_bootstrap(&ind, Some(2), Some(vec![7])).is_err());
    assert!(seq_bootstrap(&[], Some(3), None).is_err());
    assert_eq!(seq_bootstrap(&[], None, None), Ok(vec![]));
    assert_eq!(seq_bootstrap(&ind, Some(2), Some(vec![1, 0])), Ok(vec![0, 1]));

    assert_eq!(num_concurrent_events(0, &[(0, 1)], &[]), Vec::<usize>::new());
}

#[test]
fn microstructural_estimators_reject_mismatched_lengths() {
    assert!(get_roll_impact(&LONG, &SHORT, 2).is_err());
    assert!(get_corwin_schultz_estimator(&LONG, &SHORT, 2).is_err());
    assert!(get_bekker_parkinson_vol(&LONG, &SHORT, 2).is_err());
    assert!(get_bar_based_kyle_lambda(&LONG, &SHORT, 2).is_err());
    assert!(get_bar_based_amihud_lambda(&LONG, &SHORT, 2).is_err());
    assert!(get_bar_based_hasbrouck_lambda(&LONG, &SHORT, 2).is_err());
    assert!(get_vpin(&LONG, &SHORT, 2).is_err());
    assert!(get_bvc_buy_volume(&LONG, &SHORT, 2).is_err());
    assert!(get_trades_based_kyle_lambda(&LONG, &LONG, &SHORT).is_err());
    assert!(vwap(&LONG, &SHORT).is_err());
}

#[test]
fn quantile_mapping_rejects_empty_and_nan_input() {
    assert!(quantile_mapping(&[], 2).is_err());
    assert!(quantile_mapping(&[1.0, f64::NAN, 3.0], 2).is_err());
    assert!(quantile_mapping(&[1.0, 2.0, 3.0], 2).is_ok());
}

#[test]
fn plug_in_entropy_rejects_a_word_longer_than_the_message() {
    assert!(get_plug_in_entropy("11", 5).is_err());
    assert!(get_plug_in_entropy("11", 0).is_err());
    assert!(get_plug_in_entropy("", 1).is_err());
    assert!(get_plug_in_entropy("1101", 2).is_ok());
}

#[test]
fn entropy_functions_accept_the_encoders_full_alphabet() {
    // quantile_mapping hands out letters up to U+00FF; those above U+007F are two bytes in
    // UTF-8, and the entropy functions used to slice the message by byte.
    let values: Vec<f64> = (0..400).map(|i| ((i * 37) % 400) as f64).collect();
    let encoding = quantile_mapping(&values, 200).unwrap();
    let message = encode_array(&values, &encoding).unwrap();
    assert!(!message.is_ascii());

    let n = message.chars().count() as f64;
    let shannon = get_shannon_entropy(&message);
    assert!(shannon > 0.0 && shannon <= n.log2() + 1e-12, "{shannon}");
    assert!(get_plug_in_entropy(&message, 2).unwrap() > 0.0);
    let lz = get_lempel_ziv_entropy(&message);
    assert!(lz > 0.0 && lz <= 1.0, "{lz}");
    assert!(get_konto_entropy(&message, 10) > 0.0);
}

#[test]
fn entropy_of_an_ascii_message_is_unchanged_by_the_char_rewrite() {
    // Hand-computed: "aabb" has two equiprobable letters -> 1 bit; its words of length 1 over
    // the first three positions are a, a, b -> H(2/3, 1/3).
    assert!((get_shannon_entropy("aabb") - 1.0).abs() < 1e-12);
    let expected = -((2.0f64 / 3.0) * (2.0f64 / 3.0).log2() + (1.0 / 3.0) * (1.0f64 / 3.0).log2());
    assert!((get_plug_in_entropy("aabb", 1).unwrap() - expected).abs() < 1e-12);
    // Lempel-Ziv dictionary of "aabb": a, ab, b -> 3 / 4.
    assert!((get_lempel_ziv_entropy("aabb") - 0.75).abs() < 1e-12);
}

#[test]
fn ef3m_rejects_short_moments_and_unknown_variants() {
    assert_eq!(
        centered_moment(&[0.7], 5),
        Err(InputError::TooShort { name: "moments", len: 1, min: 5 })
    );
    assert!(centered_moment(&[], 0).is_err());

    let moments = vec![0.7, 2.6, 0.4, 25.0, -59.8];
    assert!(M2N::new(vec![0.7, 2.6], 1e-2, 5.0, 1, 1, 100, 1).single_fit_loop(None).is_err());
    assert!(M2N::new(moments.clone(), 1e-2, 5.0, 1, 3, 100, 1).single_fit_loop(None).is_err());
    assert!(M2N::new(moments.clone(), 1e-2, 5.0, 1, 3, 100, 1).mp_fit().is_err());
    assert!(M2N::new(moments, 1e-2, 5.0, 1, 1, 100, 1).single_fit_loop(None).is_ok());
}
