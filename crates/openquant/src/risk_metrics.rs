//! Historical (non-parametric) risk measures: portfolio variance, value at risk, expected
//! shortfall and conditional drawdown at risk.
//!
//! This module is not from AFML; it ports mlfinlab's `RiskMetrics` class. References: Artzner,
//! Delbaen, Eber and Heath (1999), *Coherent measures of risk*; Rockafellar and Uryasev (2000),
//! *Optimization of conditional value-at-risk*; Chekhlov, Uryasev and Zabarankin (2005),
//! *Drawdown measure in portfolio optimization*.
//!
//! Conventions:
//! - No distribution is fitted. Quantiles use the "higher" rule (as
//!   `numpy.quantile(..., method="higher")`): with `n` sorted values the `q`-quantile is the
//!   element at index `ceil(q * (n - 1))`, so it is always an observed value.
//! - VaR and expected shortfall take per-period **returns** and are returned **as returns, with
//!   their sign** (a 5% VaR of `-0.0166` is a 1.66% loss). Their `confidence_level` is the
//!   **lower-tail probability** (0.05, not 0.95).
//! - Conditional drawdown at risk takes a **cumulative** series (equity curve, price or
//!   cumulative return), not returns, and its `confidence_level` is the **upper-tail** level
//!   (0.95 averages the worst 5% of drawdowns).
//! - Nothing is annualised or horizon-scaled: a VaR of daily returns is a one-day VaR.
//! - The `_from_matrix` variants silently read only the **first column**.
//! - `NaN` inputs are sorted with [`f64::total_cmp`] and count toward `n`, which shifts the
//!   quantile; drop them first.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::risk_metrics::{RiskMetrics, RiskMetricsError};
//!
//! # fn main() -> Result<(), RiskMetricsError> {
//! let risk = RiskMetrics;
//! let returns = [-0.08, -0.03, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04];
//!
//! // ceil(0.25 * 9) = 3: the fourth-smallest return. The three below it average -0.04.
//! assert_eq!(risk.calculate_value_at_risk(&returns, 0.25)?, 0.0);
//! assert!((risk.calculate_expected_shortfall(&returns, 0.25)? + 0.04).abs() < 1e-12);
//!
//! // Drawdowns of this equity curve are 0 0 1 0 4 1; at 0.6 the threshold is 1 and
//! // (1, 1, 4) average 2.
//! let equity = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0];
//! assert_eq!(risk.calculate_conditional_drawdown_risk(&equity, 0.6)?, 2.0);
//!
//! let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
//! assert!((risk.calculate_variance(&covariance, &[0.6, 0.4])? - 0.0336).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

use crate::util::stats::{self, QuantileMethod};
use nalgebra::DMatrix;

/// Errors returned by [`RiskMetrics`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RiskMetricsError {
    /// The return / cumulative series (or the matrix it is read from) is empty.
    #[error("returns must not be empty")]
    EmptyInput,
    /// `confidence_level` is outside `[0, 1]` (or `NaN`).
    #[error("confidence level must be in [0, 1]")]
    InvalidConfidenceLevel,
    /// The covariance matrix is not square or its size differs from the number of weights.
    #[error("inputs have different lengths")]
    DimensionMismatch,
}

/// Stateless calculator of historical risk measures; every measure is a method.
///
/// Construct it as the unit value `RiskMetrics` (or `RiskMetrics::default()`). See the
/// [module documentation](self) for the sign and confidence-level conventions.
#[derive(Debug, Default, Clone)]
pub struct RiskMetrics;

impl RiskMetrics {
    /// Portfolio variance `w' Σ w` for weights `weights` and covariance matrix `covariance`.
    ///
    /// `covariance` is `n x n` with rows/columns in the same asset order as `weights`; the
    /// result is in the units of the covariance, e.g. per-period variance of returns. Weights
    /// are used as given (not normalised). Empty inputs (`0 x 0` and `[]`) return `Ok(0.0)`.
    ///
    /// # Errors
    ///
    /// [`RiskMetricsError::DimensionMismatch`] if `covariance` is not square or its size differs
    /// from `weights.len()`.
    ///
    /// ```
    /// use nalgebra::DMatrix;
    /// use openquant::risk_metrics::{RiskMetrics, RiskMetricsError};
    ///
    /// let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
    /// // 0.36 * 0.04 + 2 * 0.24 * 0.01 + 0.16 * 0.09 = 0.0336
    /// let var = RiskMetrics.calculate_variance(&covariance, &[0.6, 0.4]).unwrap();
    /// assert!((var - 0.0336).abs() < 1e-12);
    /// assert_eq!(
    ///     RiskMetrics.calculate_variance(&covariance, &[1.0]),
    ///     Err(RiskMetricsError::DimensionMismatch)
    /// );
    /// ```
    pub fn calculate_variance(
        &self,
        covariance: &DMatrix<f64>,
        weights: &[f64],
    ) -> Result<f64, RiskMetricsError> {
        if covariance.nrows() != covariance.ncols() || covariance.nrows() != weights.len() {
            return Err(RiskMetricsError::DimensionMismatch);
        }

        let mut total = 0.0;
        for i in 0..weights.len() {
            for j in 0..weights.len() {
                total += weights[i] * covariance[(i, j)] * weights[j];
            }
        }
        Ok(total)
    }

    /// Historical value at risk: the "higher" `confidence_level`-quantile of `returns`.
    ///
    /// `returns` are per-period returns in any order; the result is a return with its sign
    /// (typically negative). `confidence_level` is the **lower-tail probability** in `[0, 1]`
    /// (0.05 for a 5% VaR). The quantile is the sorted element at index
    /// `ceil(confidence_level * (n - 1))`, never interpolated, so with 20 returns the 5% VaR is
    /// the second-worst.
    ///
    /// # Errors
    ///
    /// - [`RiskMetricsError::InvalidConfidenceLevel`] if `confidence_level` is outside `[0, 1]`.
    /// - [`RiskMetricsError::EmptyInput`] if `returns` is empty.
    ///
    /// ```
    /// use openquant::risk_metrics::RiskMetrics;
    ///
    /// let returns = [-0.08, -0.03, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04];
    /// assert_eq!(RiskMetrics.calculate_value_at_risk(&returns, 0.25).unwrap(), 0.0);
    /// assert_eq!(RiskMetrics.calculate_value_at_risk(&returns, 0.0).unwrap(), -0.08);
    /// ```
    pub fn calculate_value_at_risk(
        &self,
        returns: &[f64],
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        quantile_higher(returns, confidence_level)
    }

    /// [`Self::calculate_value_at_risk`] on the **first column** of `returns`; other columns
    /// are ignored.
    ///
    /// # Errors
    ///
    /// - [`RiskMetricsError::EmptyInput`] if `returns` has no rows or no columns.
    /// - [`RiskMetricsError::InvalidConfidenceLevel`] if `confidence_level` is outside `[0, 1]`.
    pub fn calculate_value_at_risk_from_matrix(
        &self,
        returns: &DMatrix<f64>,
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        self.calculate_value_at_risk(&first_col(returns)?, confidence_level)
    }

    /// Historical expected shortfall (conditional VaR): the mean of the returns **strictly
    /// below** [`Self::calculate_value_at_risk`] at the same `confidence_level`.
    ///
    /// Units and `confidence_level` follow [`Self::calculate_value_at_risk`] (lower-tail
    /// probability; the result is a signed return). If no return lies strictly below the VaR —
    /// for example constant returns or `confidence_level == 0` — the result is `Ok(NaN)`, not
    /// an error.
    ///
    /// # Errors
    ///
    /// - [`RiskMetricsError::InvalidConfidenceLevel`] if `confidence_level` is outside `[0, 1]`.
    /// - [`RiskMetricsError::EmptyInput`] if `returns` is empty.
    ///
    /// ```
    /// use openquant::risk_metrics::RiskMetrics;
    ///
    /// let returns = [-0.08, -0.03, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04];
    /// // VaR is 0.0; the three returns below it average -0.04.
    /// let es = RiskMetrics.calculate_expected_shortfall(&returns, 0.25).unwrap();
    /// assert!((es + 0.04).abs() < 1e-12);
    /// // Nothing lies strictly below the minimum, so the tail is empty.
    /// assert!(RiskMetrics.calculate_expected_shortfall(&returns, 0.0).unwrap().is_nan());
    /// ```
    pub fn calculate_expected_shortfall(
        &self,
        returns: &[f64],
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        let var = self.calculate_value_at_risk(returns, confidence_level)?;
        let tail: Vec<f64> = returns.iter().copied().filter(|v| *v < var).collect();
        if tail.is_empty() {
            return Ok(f64::NAN);
        }
        Ok(tail.iter().sum::<f64>() / tail.len() as f64)
    }

    /// [`Self::calculate_expected_shortfall`] on the **first column** of `returns`; other
    /// columns are ignored.
    ///
    /// # Errors
    ///
    /// - [`RiskMetricsError::EmptyInput`] if `returns` has no rows or no columns.
    /// - [`RiskMetricsError::InvalidConfidenceLevel`] if `confidence_level` is outside `[0, 1]`.
    pub fn calculate_expected_shortfall_from_matrix(
        &self,
        returns: &DMatrix<f64>,
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        self.calculate_expected_shortfall(&first_col(returns)?, confidence_level)
    }

    /// Conditional drawdown at risk (Chekhlov, Uryasev and Zabarankin, 2005): the mean of the
    /// worst `1 - confidence_level` share of the drawdown series.
    ///
    /// `cumulative` must be a cumulative series — an equity curve, a price or a cumulative
    /// return — not per-period returns: the drawdown is `running_max(cumulative) - cumulative`,
    /// in the units of the input. The threshold is the "higher" `confidence_level`-quantile of
    /// the drawdowns, and every drawdown at or above it is averaged, so the tail is never empty.
    ///
    /// `confidence_level` is the **upper-tail** level here: 0.95 averages the worst 5% of
    /// drawdowns. This is the opposite of [`Self::calculate_value_at_risk`] and
    /// [`Self::calculate_expected_shortfall`], which take the lower-tail probability (0.05).
    ///
    /// On finite input the result is never `NaN`; a `NaN` in `cumulative` can make it `NaN`.
    ///
    /// # Errors
    ///
    /// - [`RiskMetricsError::InvalidConfidenceLevel`] if `confidence_level` is outside `[0, 1]`.
    /// - [`RiskMetricsError::EmptyInput`] if `cumulative` is empty.
    ///
    /// ```
    /// use openquant::risk_metrics::RiskMetrics;
    ///
    /// // Drawdowns 0 0 1 0 4 1.
    /// let equity = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0];
    /// let cdar = |a| RiskMetrics.calculate_conditional_drawdown_risk(&equity, a).unwrap();
    /// assert_eq!(cdar(0.6), 2.0); // threshold 1: (1, 1, 4) average 2
    /// assert_eq!(cdar(0.9), 4.0); // only the maximum drawdown
    /// ```
    pub fn calculate_conditional_drawdown_risk(
        &self,
        cumulative: &[f64],
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        validate_confidence(confidence_level)?;
        if cumulative.is_empty() {
            return Err(RiskMetricsError::EmptyInput);
        }

        let mut running_max = f64::NEG_INFINITY;
        let mut drawdown = Vec::with_capacity(cumulative.len());
        for &v in cumulative {
            running_max = running_max.max(v);
            drawdown.push(running_max - v);
        }

        let q = quantile_higher(&drawdown, confidence_level)?;
        let tail: Vec<f64> = drawdown.into_iter().filter(|v| *v >= q).collect();
        if tail.is_empty() {
            // only reachable when q is NaN (NaN in the input)
            return Ok(f64::NAN);
        }
        Ok(tail.iter().sum::<f64>() / tail.len() as f64)
    }

    /// [`Self::calculate_conditional_drawdown_risk`] on the first column of `cumulative`.
    ///
    /// # Errors
    ///
    /// - [`RiskMetricsError::EmptyInput`] if `cumulative` has no rows or no columns.
    /// - [`RiskMetricsError::InvalidConfidenceLevel`] if `confidence_level` is outside `[0, 1]`.
    pub fn calculate_conditional_drawdown_risk_from_matrix(
        &self,
        cumulative: &DMatrix<f64>,
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        self.calculate_conditional_drawdown_risk(&first_col(cumulative)?, confidence_level)
    }
}

fn validate_confidence(confidence_level: f64) -> Result<(), RiskMetricsError> {
    if !(0.0..=1.0).contains(&confidence_level) {
        return Err(RiskMetricsError::InvalidConfidenceLevel);
    }
    Ok(())
}

fn quantile_higher(values: &[f64], q: f64) -> Result<f64, RiskMetricsError> {
    validate_confidence(q)?;
    stats::quantile(values, q, QuantileMethod::Higher).ok_or(RiskMetricsError::EmptyInput)
}

fn first_col(m: &DMatrix<f64>) -> Result<Vec<f64>, RiskMetricsError> {
    if m.nrows() == 0 || m.ncols() == 0 {
        return Err(RiskMetricsError::EmptyInput);
    }
    Ok((0..m.nrows()).map(|r| m[(r, 0)]).collect())
}
