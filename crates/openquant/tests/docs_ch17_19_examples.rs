//! The Rust examples on the structural_breaks and microstructural_features docs pages, run as
//! tests. `check:examples` only compiles page snippets; these execute them.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn structural_breaks_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::structural_breaks::{
        get_chow_type_stat, get_sadf, SadfLags, StructuralBreakError,
    };

    // A log price whose increments compound: explosive by construction.
    let mut y = vec![4.0_f64];
    for t in 1..120 {
        let wobble = if t % 2 == 0 { 0.002 } else { -0.002 };
        y.push(y[t - 1] + 0.0005 * 1.05_f64.powi(t as i32) + wobble);
    }

    let sadf = get_sadf(&y, "linear", true, 20, SadfLags::Fixed(1))?;
    // One lag uses two leading bars; the first statistic then needs min_length more.
    assert_eq!(sadf.len(), y.len() - 2 - 20);
    assert!(sadf.last().unwrap() > &3.0);

    let chow = get_chow_type_stat(&y, 20)?;
    assert_eq!(chow.len(), y.len() - 2 * 20);

    // Too short a series is not an error for these two: they return nothing.
    assert!(get_chow_type_stat(&y[..30], 20)?.is_empty());
    assert!(matches!(
        get_sadf(&y, "cubic", true, 20, SadfLags::Fixed(1)),
        Err(StructuralBreakError::InvalidModel(_))
    ));
    Ok(())
}

#[test]
fn microstructural_features_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::microstructural_features::{
        encode_tick_rule_array, get_lempel_ziv_entropy, get_plug_in_entropy, get_shannon_entropy,
        get_trades_based_kyle_lambda, get_vpin, vwap,
    };

    // Regression through the origin of price change on signed volume: sum(xy) / sum(x^2).
    let lambda =
        get_trades_based_kyle_lambda(&[0.2, -0.1, 0.4], &[10.0, 5.0, 20.0], &[1.0, -1.0, 1.0])?;
    assert!((lambda - 0.02).abs() < 1e-12);

    // VPIN over 3 bars of equal volume: mean |buys - sells| / volume. NaN until the window fills.
    let vpin = get_vpin(&[100.0; 5], &[80.0, 20.0, 50.0, 90.0, 10.0], 3)?;
    assert!(vpin[0].is_nan() && vpin[1].is_nan());
    assert!((vpin[2] - 0.4).abs() < 1e-12);

    assert!((vwap(&[1000.0, 2000.0], &[10.0, 10.0])? - 150.0).abs() < 1e-12);

    // A perfectly periodic message: maximal Shannon entropy, low entropy by every other measure.
    assert_eq!(encode_tick_rule_array(&[1, 1, -1, 0])?, "aabc");
    assert_eq!(get_shannon_entropy("abababab"), 1.0);
    assert_eq!(get_plug_in_entropy("abababab", 2)?, 0.5);
    assert_eq!(get_lempel_ziv_entropy("abababab"), 0.5);
    Ok(())
}
