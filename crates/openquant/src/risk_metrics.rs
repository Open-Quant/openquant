use nalgebra::DMatrix;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RiskMetricsError {
    #[error("returns must not be empty")]
    EmptyInput,
    #[error("confidence level must be in [0, 1]")]
    InvalidConfidenceLevel,
    #[error("inputs have different lengths")]
    DimensionMismatch,
}

#[derive(Debug, Default, Clone)]
pub struct RiskMetrics;

impl RiskMetrics {
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

    pub fn calculate_value_at_risk(
        &self,
        returns: &[f64],
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        quantile_higher(returns, confidence_level)
    }

    pub fn calculate_value_at_risk_from_matrix(
        &self,
        returns: &DMatrix<f64>,
        confidence_level: f64,
    ) -> Result<f64, RiskMetricsError> {
        self.calculate_value_at_risk(&first_col(returns)?, confidence_level)
    }

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
    if values.is_empty() {
        return Err(RiskMetricsError::EmptyInput);
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let pos = (q * (n.saturating_sub(1) as f64)).ceil() as usize;
    Ok(sorted[pos.min(n - 1)])
}

fn first_col(m: &DMatrix<f64>) -> Result<Vec<f64>, RiskMetricsError> {
    if m.nrows() == 0 || m.ncols() == 0 {
        return Err(RiskMetricsError::EmptyInput);
    }
    Ok((0..m.nrows()).map(|r| m[(r, 0)]).collect())
}
