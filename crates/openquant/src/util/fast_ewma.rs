use super::InputError;

/// Exponentially weighted moving average with span-like `window`.
/// Mirrors mlfinlab.util.fast_ewma.ewma.
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
