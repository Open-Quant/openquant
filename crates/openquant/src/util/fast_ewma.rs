//! Span-style exponentially weighted moving average, matching `mlfinlab.util.fast_ewma`.
//!
//! One function, [`ewma`], with decay `alpha = 2 / (window + 1)` and the bias-corrected
//! ("adjusted") weighting of pandas' `ewm(span=window, adjust=True).mean()`: each output is the
//! weighted mean of every value so far, with weight `(1 - alpha)^k` on the value `k` steps back.
//! Early outputs are therefore not dragged toward the first value, and `window` is a span, not a
//! hard lookback: no past value ever gets exactly zero weight.
//!
//! It is the shared decay convention behind daily volatility (AFML Snippet 3.1, in
//! [`crate::util::volatility`]) and the EWMA-derived features and thresholds elsewhere in the
//! crate.
//!
//! ```
//! use openquant::util::fast_ewma::ewma;
//!
//! // window 3 gives alpha = 0.5: 5 = (0.5*3 + 6) / 1.5 and 9 = (0.25*3 + 0.5*6 + 12) / 1.75.
//! assert_eq!(ewma(&[3.0, 6.0, 12.0], 3).unwrap(), vec![3.0, 5.0, 9.0]);
//! ```
#![deny(missing_docs)]

use super::InputError;

/// Returns the bias-corrected exponentially weighted moving average of `arr_in` with span
/// `window` (the `mlfinlab.util.fast_ewma.ewma` recursion).
///
/// `arr_in` is ordered oldest first. Output `t` is
/// `Σ_{k=0..=t} (1-α)^k x_{t-k} / Σ_{k=0..=t} (1-α)^k` with `α = 2 / (window + 1)`, so the first
/// output equals the first input and the output has the same length as the input. An empty
/// input gives an empty output. `window = 1` gives `α = 1`, which returns the input unchanged.
///
/// Values are not validated: a `NaN` input makes that output and every later one `NaN` (unlike
/// pandas, which skips missing values by default).
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `window` is zero.
///
/// ```
/// use openquant::util::fast_ewma::ewma;
///
/// assert_eq!(ewma(&[1.0, 4.0, 2.0], 1).unwrap(), vec![1.0, 4.0, 2.0]);
/// assert!(ewma(&[], 5).unwrap().is_empty());
/// ```
pub fn ewma(arr_in: &[f64], window: usize) -> Result<Vec<f64>, InputError> {
    if window == 0 {
        return Err(InputError::OutOfRange {
            name: "window",
            value: 0.0,
            expected: "a positive integer",
        });
    }
    if arr_in.is_empty() {
        return Ok(Vec::new());
    }

    let alpha = 2.0 / (window as f64 + 1.0);
    let mut weight = 1.0;
    let mut ewma_old = arr_in[0];
    let mut out = vec![0.0; arr_in.len()];
    out[0] = ewma_old;

    for i in 1..arr_in.len() {
        weight += (1.0 - alpha).powi(i as i32);
        ewma_old = ewma_old * (1.0 - alpha) + arr_in[i];
        out[i] = ewma_old / weight;
    }
    Ok(out)
}
