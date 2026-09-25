//! Fixed-step resampling of a price matrix (rows are dates, columns are assets).
#![deny(missing_docs)]

use nalgebra::DMatrix;

/// Rows per period for a resampling code: 5 for weekly, 21 for monthly, otherwise 1.
pub(crate) fn freq_step(resample_by: Option<&str>) -> usize {
    match resample_by.map(str::to_ascii_lowercase).as_deref() {
        Some("w" | "week" | "weekly") => 5,
        Some("m" | "month" | "monthly") => 21,
        _ => 1,
    }
}

/// Keep the last row of every block of `step` rows. A series shorter than one block is returned
/// unchanged.
pub(crate) fn resample_prices(prices: &DMatrix<f64>, step: usize) -> DMatrix<f64> {
    if step <= 1 {
        return prices.clone_owned();
    }
    let kept: Vec<usize> = (step - 1..prices.nrows()).step_by(step).collect();
    if kept.is_empty() {
        return prices.clone_owned();
    }
    DMatrix::from_fn(kept.len(), prices.ncols(), |r, c| prices[(kept[r], c)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_whole_rows_in_order() {
        // Row r holds [r, 100 + r, 200 + r], so a scrambled layout is visible in every cell.
        let prices = DMatrix::from_fn(12, 3, |r, c| (100 * c + r) as f64);
        let weekly = resample_prices(&prices, 5);
        assert_eq!(weekly.nrows(), 2);
        assert_eq!(weekly.row(0).iter().copied().collect::<Vec<_>>(), vec![4.0, 104.0, 204.0]);
        assert_eq!(weekly.row(1).iter().copied().collect::<Vec<_>>(), vec![9.0, 109.0, 209.0]);
    }

    #[test]
    fn short_series_and_unit_step_are_unchanged() {
        let prices = DMatrix::from_fn(3, 2, |r, c| (10 * c + r) as f64);
        assert_eq!(resample_prices(&prices, 5), prices);
        assert_eq!(resample_prices(&prices, 1), prices);
    }

    #[test]
    fn frequency_codes() {
        assert_eq!(freq_step(Some("W")), 5);
        assert_eq!(freq_step(Some("monthly")), 21);
        assert_eq!(freq_step(Some("B")), 1);
        assert_eq!(freq_step(Some("unknown")), 1);
        assert_eq!(freq_step(None), 1);
    }
}
