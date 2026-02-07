/// Exponentially weighted moving average with span-like `window`.
/// Mirrors mlfinlab.util.fast_ewma.ewma.
pub fn ewma(arr_in: &[f64], window: usize) -> Vec<f64> {
    if arr_in.is_empty() {
        return Vec::new();
    }
    assert!(window > 0, "window must be > 0");

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
    out
}
