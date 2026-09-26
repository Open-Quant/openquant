//! EF3M: fit a mixture of two Gaussians by matching its raw moments exactly.
//!
//! The mixture `p_1 N(mu_1, sigma_1^2) + (1 - p_1) N(mu_2, sigma_2^2)` has five parameters,
//! so its first five raw moments pin it down. EF3M, the *Exact Fit of the first 3 Moments*
//! (López de Prado and Foreman, 2014, *Quantitative Finance* 14(5), 913–930), solves the
//! moment equations iteratively from a grid of starting values of `mu_2`, each with a random
//! starting `p_1`, and keeps the iterate with the smallest squared moment error. AFML uses
//! the fitted mixture in §10.2 (strategy-independent bet sizing, see
//! [`crate::bet_sizing::bet_size_reserve`]) and §15.4.1 (the average loss and gain of a
//! strategy, see [`crate::strategy_risk`]).
//!
//! - [`M2N`] holds the moments and settings; [`M2N::single_fit_loop`] is one search and
//!   [`M2N::mp_fit`] repeats it `n_runs` times.
//! - [`most_likely_parameters`] takes, column by column, the mode of a kernel density over
//!   the runs, the paper's way of summarising the random fits.
//! - [`centered_moment`] and [`raw_moment`] convert between raw and centred moments.
//!
//! Conventions: `moments` are **raw** moments `E[x^k]` for `k = 1..=5`, in that order.
//! Parameter vectors are ordered `[mu_1, mu_2, sigma_1, sigma_2, p_1]` (sigmas are standard
//! deviations), the order [`crate::bet_sizing`] expects. `variant` 1 fits four moments and
//! variant 2 five; variant 2 is more accurate and runs about the mean (see
//! [`M2N::single_fit_loop`]). Fits are random and unseeded (the starting `p_1` comes from
//! the thread-local RNG); use several runs and [`most_likely_parameters`], and check that
//! the runs agree on which component is which, since the labels can come back swapped. The
//! five modes need not come from the same run, nor reproduce the moments. [`M2N::mp_fit`] is
//! serial; the deprecated `num_workers` field is ignored.
//!
//! ```
//! use openquant::ef3m::{centered_moment, M2N};
//!
//! // Exact raw moments of 0.7 N(-1, 1) + 0.3 N(2, 0.5^2).
//! let truth = [-1.0, 2.0, 1.0, 0.5, 0.7];
//! let moments = M2N::with_defaults(vec![]).get_moments(&truth, true).unwrap();
//! assert!((moments[0] + 0.1).abs() < 1e-12); // 0.7 * -1 + 0.3 * 2
//! let variance = centered_moment(&moments, 2).unwrap();
//! assert!((variance - 2.665).abs() < 1e-12);
//!
//! // Started at the true mu_2 and p_1, one iteration of either variant reproduces the
//! // mixture: the truth is a fixed point.
//! let m2n = M2N::with_defaults(moments.clone());
//! for step in [m2n.iter_4(2.0, 0.7).unwrap(), m2n.iter_5(2.0, 0.7).unwrap()] {
//!     assert!(step.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-9));
//! }
//! ```
#![deny(missing_docs)]

use crate::util::InputError;
use rand::Rng;
use std::collections::{BTreeMap, HashSet};

/// An EF3M fitter for a mixture of two Gaussians: the target moments, the search settings,
/// and the best fit found so far.
///
/// Build it with [`M2N::new`] or [`M2N::with_defaults`] and run [`M2N::mp_fit`] or
/// [`M2N::single_fit_loop`]. Parameter vectors are `[mu_1, mu_2, sigma_1, sigma_2, p_1]`.
#[derive(Debug, Clone)]
pub struct M2N {
    /// Target raw moments `E[x^k]`, `k = 1..=5`. The fit needs all five.
    pub moments: Vec<f64>,
    /// Convergence tolerance on `p_1`; also the spacing of the `mu_2` start grid, which has
    /// about `1 / epsilon` points. Must be finite and `> 0` ([`M2N::single_fit_loop`] checks).
    pub epsilon: f64,
    /// Width of the `mu_2` start grid in standard deviations: starts run from
    /// `m_1 + epsilon * factor * sigma` to about `m_1 + factor * sigma`.
    pub factor: f64,
    /// Number of independent searches [`M2N::mp_fit`] runs.
    pub n_runs: usize,
    /// `1` to fit four moments ([`M2N::iter_4`]) or `2` to fit five ([`M2N::iter_5`]).
    pub variant: usize,
    /// Maximum iterations of one attempt in [`M2N::fit`]; `0` runs none.
    pub max_iter: usize,
    /// Ignored: [`M2N::mp_fit`] runs serially. Kept only so existing code still compiles.
    #[deprecated(note = "ignored: M2N::mp_fit runs serially")]
    pub num_workers: isize,
    /// Scratch: the moments implied by the last iterate of [`M2N::fit`] (or by
    /// [`M2N::get_moments`] with `return_result = false`).
    pub new_moments: Vec<f64>,
    /// Best parameters found so far, `[mu_1, mu_2, sigma_1, sigma_2, p_1]`; all zeros until
    /// an admissible iterate improves on the initial error.
    pub parameters: Vec<f64>,
    /// Squared moment error of `parameters`; starts at the sum of squared target moments.
    pub error: f64,
}

/// One fitted mixture, a row of the output of [`M2N::single_fit_loop`] and [`M2N::mp_fit`].
#[derive(Debug, Clone)]
pub struct FitResultRow {
    /// Mean of component 1.
    pub mu_1: f64,
    /// Mean of component 2.
    pub mu_2: f64,
    /// Standard deviation of component 1.
    pub sigma_1: f64,
    /// Standard deviation of component 2.
    pub sigma_2: f64,
    /// Weight of component 1, in `[0, 1]`; component 2 has weight `1 - p_1`.
    pub p_1: f64,
    /// Sum of squared differences between the target raw moments and the five raw moments
    /// implied by this row's parameters.
    pub error: f64,
}

fn comb(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k_eff = k.min(n - k);
    let mut num = 1.0;
    let mut den = 1.0;
    for i in 0..k_eff {
        num *= (n - i) as f64;
        den *= (i + 1) as f64;
    }
    num / den
}

fn round_to_5(x: f64) -> f64 {
    (x * 100_000.0).round() / 100_000.0
}

impl M2N {
    /// Creates a fitter; see the field docs for each argument. Nothing is validated here:
    /// [`M2N::single_fit_loop`] and [`M2N::mp_fit`] check `moments` and `variant`.
    /// `num_workers` is ignored ([`M2N::mp_fit`] runs serially).
    ///
    /// `parameters` starts at zeros and `error` at the sum of squared `moments`.
    pub fn new(
        moments: Vec<f64>,
        epsilon: f64,
        factor: f64,
        n_runs: usize,
        variant: usize,
        max_iter: usize,
        num_workers: isize,
    ) -> Self {
        let error = moments.iter().map(|m| m * m).sum();
        #[allow(deprecated)]
        Self {
            moments,
            epsilon,
            factor,
            n_runs,
            variant,
            max_iter,
            num_workers,
            new_moments: vec![0.0; 5],
            parameters: vec![0.0; 5],
            error,
        }
    }

    /// Creates a fitter with the defaults `epsilon = 1e-5`, `factor = 5`, `n_runs = 1`,
    /// `variant = 2` (five moments, the more accurate variant) and `max_iter = 100_000`.
    pub fn with_defaults(moments: Vec<f64>) -> Self {
        Self::new(moments, 1e-5, 5.0, 1, 2, 100_000, -1)
    }

    /// The first five raw moments `E[x^k]`, `k = 1..=5`, of the mixture with the given
    /// parameters `[mu_1, mu_2, sigma_1, sigma_2, p_1]`.
    ///
    /// With `return_result = true` the moments are returned; with `false` they are stored in
    /// [`M2N::new_moments`] and `None` is returned. Taking exactly five parameters by type
    /// means a short slice cannot reach the indexing (convert a slice with
    /// `slice.try_into()`); the values themselves are not validated.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn get_moments(&mut self, parameters: &[f64; 5], return_result: bool) -> Option<Vec<f64>> {
        let [u_1, u_2, s_1, s_2, p_1] = *parameters;
        let p_2 = 1.0 - p_1;

        let m_1 = p_1 * u_1 + p_2 * u_2;
        let m_2 = p_1 * (s_1.powi(2) + u_1.powi(2)) + p_2 * (s_2.powi(2) + u_2.powi(2));
        let m_3 = p_1 * (3.0 * s_1.powi(2) * u_1 + u_1.powi(3))
            + p_2 * (3.0 * s_2.powi(2) * u_2 + u_2.powi(3));
        let m_4 = p_1 * (3.0 * s_1.powi(4) + 6.0 * s_1.powi(2) * u_1.powi(2) + u_1.powi(4))
            + p_2 * (3.0 * s_2.powi(4) + 6.0 * s_2.powi(2) * u_2.powi(2) + u_2.powi(4));
        let m_5 = p_1 * (15.0 * s_1.powi(4) * u_1 + 10.0 * s_1.powi(2) * u_1.powi(3) + u_1.powi(5))
            + p_2 * (15.0 * s_2.powi(4) * u_2 + 10.0 * s_2.powi(2) * u_2.powi(3) + u_2.powi(5));
        let out = vec![m_1, m_2, m_3, m_4, m_5];

        if return_result {
            Some(out)
        } else {
            self.new_moments = out;
            None
        }
    }

    /// One step of the four-moment variant (variant 1): from a guess of `mu_2` and `p_1`,
    /// solves `mu_1` from `m_1`, `sigma_2` from `m_3`, `sigma_1` from `m_2`, and a new `p_1`
    /// from `m_4`.
    ///
    /// Returns `[mu_1, mu_2, sigma_1, sigma_2, p_1_new]` (with `mu_2` unchanged), or an empty
    /// vector if the step is inadmissible: a zero denominator, a negative variance, or a new
    /// `p_1` outside `[0, 1]` (NaN included).
    ///
    /// # Errors
    ///
    /// [`InputError::TooShort`] if [`M2N::moments`] has fewer than four entries.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn iter_4(&self, mu_2: f64, p_1: f64) -> Result<Vec<f64>, InputError> {
        self.require_moments(4)?;
        Ok(self.iter_4_unchecked(mu_2, p_1))
    }

    fn iter_4_unchecked(&self, mu_2: f64, p_1: f64) -> Vec<f64> {
        let m_1 = self.moments[0];
        let m_2 = self.moments[1];
        let m_3 = self.moments[2];
        let m_4 = self.moments[3];

        let mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1;
        let den_24 = 3.0 * (1.0 - p_1) * (mu_2 - mu_1);
        if den_24 == 0.0 {
            return vec![];
        }
        let sigma_2_squared = (m_3 + 2.0 * p_1 * mu_1.powi(3) + (p_1 - 1.0) * mu_2.powi(3)
            - 3.0 * mu_1 * (m_2 + mu_2.powi(2) * (p_1 - 1.0)))
            / den_24;
        if sigma_2_squared < 0.0 {
            return vec![];
        }
        let sigma_2 = sigma_2_squared.sqrt();

        let sigma_1_squared =
            ((m_2 - sigma_2.powi(2) - mu_2.powi(2)) / p_1) + sigma_2.powi(2) + mu_2.powi(2)
                - mu_1.powi(2);
        if sigma_1_squared < 0.0 {
            return vec![];
        }
        let sigma_1 = sigma_1_squared.sqrt();

        let p_1_deno = 3.0 * (sigma_1.powi(4) - sigma_2.powi(4))
            + 6.0 * (sigma_1.powi(2) * mu_1.powi(2) - sigma_2.powi(2) * mu_2.powi(2))
            + mu_1.powi(4)
            - mu_2.powi(4);
        if p_1_deno == 0.0 {
            return vec![];
        }
        let p_1_new =
            (m_4 - 3.0 * sigma_2.powi(4) - 6.0 * sigma_2.powi(2) * mu_2.powi(2) - mu_2.powi(4))
                / p_1_deno;
        if !(0.0..=1.0).contains(&p_1_new) {
            return vec![];
        }

        vec![mu_1, mu_2, sigma_1, sigma_2, p_1_new]
    }

    /// One step of the five-moment variant. Like the paper, it takes the positive root for mu_2,
    /// so on raw moments it cannot return mu_2 < 0; `single_fit_loop` therefore runs this variant
    /// on centred moments (see `centred_fit_loop`).
    ///
    /// From a guess of `mu_2` and `p_1`, solves `mu_1` from `m_1`, `sigma_2` from `m_3`,
    /// `sigma_1` from `m_2`, a new `mu_2` from `m_4` (positive root) and a new `p_1` from
    /// `m_5`. Returns `[mu_1, mu_2_new, sigma_1, sigma_2, p_1_new]`, or an empty vector if the
    /// step is inadmissible: a zero denominator, a negative variance or `mu_2^2`,
    /// `1 - p_1 < 1e-4`, or a new `p_1` outside `[0, 1]` (NaN included). Called directly on
    /// raw moments it still takes the positive root (#115).
    ///
    /// # Errors
    ///
    /// [`InputError::TooShort`] if [`M2N::moments`] has fewer than five entries.
    ///
    /// # Examples
    ///
    /// See the [module documentation](self).
    pub fn iter_5(&self, mu_2: f64, p_1: f64) -> Result<Vec<f64>, InputError> {
        self.require_moments(5)?;
        Ok(self.iter_5_unchecked(mu_2, p_1))
    }

    fn iter_5_unchecked(&self, mu_2: f64, p_1: f64) -> Vec<f64> {
        let m_1 = self.moments[0];
        let m_2 = self.moments[1];
        let m_3 = self.moments[2];
        let m_4 = self.moments[3];
        let m_5 = self.moments[4];

        let mu_1 = (m_1 - (1.0 - p_1) * mu_2) / p_1;
        let den_24 = 3.0 * (1.0 - p_1) * (mu_2 - mu_1);
        if den_24 == 0.0 {
            return vec![];
        }
        let sigma_2_squared = (m_3 + 2.0 * p_1 * mu_1.powi(3) + (p_1 - 1.0) * mu_2.powi(3)
            - 3.0 * mu_1 * (m_2 + mu_2.powi(2) * (p_1 - 1.0)))
            / den_24;
        if sigma_2_squared < 0.0 {
            return vec![];
        }
        let sigma_2 = sigma_2_squared.sqrt();

        let sigma_1_squared =
            ((m_2 - sigma_2.powi(2) - mu_2.powi(2)) / p_1) + sigma_2.powi(2) + mu_2.powi(2)
                - mu_1.powi(2);
        if sigma_1_squared < 0.0 {
            return vec![];
        }
        let sigma_1 = sigma_1_squared.sqrt();

        if (1.0 - p_1) < 1e-4 {
            return vec![];
        }
        let a_1_squared = 6.0 * sigma_2.powi(4)
            + (m_4
                - p_1
                    * (3.0 * sigma_1.powi(4)
                        + 6.0 * sigma_1.powi(2) * mu_1.powi(2)
                        + mu_1.powi(4)))
                / (1.0 - p_1);
        if a_1_squared < 0.0 {
            return vec![];
        }
        let a_1 = a_1_squared.sqrt();
        let mu_2_squared = a_1 - 3.0 * sigma_2.powi(2);
        if !mu_2_squared.is_finite() || mu_2_squared < 0.0 {
            return vec![];
        }
        let mu_2_new = mu_2_squared.sqrt();

        let a_2 =
            15.0 * sigma_1.powi(4) * mu_1 + 10.0 * sigma_1.powi(2) * mu_1.powi(3) + mu_1.powi(5);
        let b_2 = 15.0 * sigma_2.powi(4) * mu_2_new
            + 10.0 * sigma_2.powi(2) * mu_2_new.powi(3)
            + mu_2_new.powi(5);
        if (a_2 - b_2) == 0.0 {
            return vec![];
        }
        let p_1_new = (m_5 - b_2) / (a_2 - b_2);
        if !(0.0..=1.0).contains(&p_1_new) {
            return vec![];
        }

        vec![mu_1, mu_2_new, sigma_1, sigma_2, p_1_new]
    }

    /// One EF3M attempt from the starting `mu_2` and a random starting `p_1` in `[0, 1)`.
    ///
    /// Iterates [`M2N::iter_4`] or [`M2N::iter_5`] (by [`M2N::variant`]) until `p_1` moves by
    /// less than [`M2N::epsilon`], a step is inadmissible, or [`M2N::max_iter`] iterations
    /// have run. Whenever an iterate's squared moment error (over the first five
    /// target moments) beats
    /// [`M2N::error`], it replaces [`M2N::parameters`] and [`M2N::error`], so they always
    /// hold the best iterate seen across calls, not the last one. The attempt runs on
    /// [`M2N::moments`] as given (for variant 2, on raw moments it cannot return
    /// `mu_2 < 0`); [`M2N::single_fit_loop`] is the entry point that centres them.
    ///
    /// # Errors
    ///
    /// - [`InputError::OutOfRange`] if `variant` is not 1 or 2.
    /// - [`InputError::TooShort`] if [`M2N::moments`] has fewer than four entries (variant 1)
    ///   or five (variant 2).
    ///
    /// Inadmissible steps and hitting `max_iter` end the attempt with `Ok(())`.
    pub fn fit(&mut self, mu_2: f64) -> Result<(), InputError> {
        match self.variant {
            1 => self.require_moments(4)?,
            2 => self.require_moments(5)?,
            _ => {
                return Err(InputError::OutOfRange {
                    name: "variant",
                    value: self.variant as f64,
                    expected: "1 (four moments) or 2 (five moments)",
                })
            }
        }
        let p_1 = rand::thread_rng().gen_range(0.0..1.0);
        self.iterate_from(mu_2, p_1);
        Ok(())
    }

    /// The iteration of [`M2N::fit`] from a given start; `moments` and `variant` must already
    /// have been checked. Returns the number of iterations run, at most `max_iter`.
    fn iterate_from(&mut self, mut mu_2: f64, mut p_1: f64) -> usize {
        let mut num_iter = 0usize;
        while num_iter < self.max_iter {
            num_iter += 1;
            // Lengths and variant were checked by the caller.
            let step = if self.variant == 1 {
                self.iter_4_unchecked(mu_2, p_1)
            } else {
                self.iter_5_unchecked(mu_2, p_1)
            };
            let Ok(parameters) = <[f64; 5]>::try_from(step) else {
                // An inadmissible step (the empty vector) ends the attempt.
                break;
            };
            let _ = self.get_moments(&parameters, false);
            let error: f64 = self
                .moments
                .iter()
                .zip(self.new_moments.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if error < self.error {
                self.parameters = parameters.to_vec();
                self.error = error;
            }

            // Stop at convergence. `self.parameters` already holds the best iterate, the one
            // `self.error` describes; the last iterate is not necessarily it.
            if (p_1 - parameters[4]).abs() < self.epsilon {
                break;
            }
            p_1 = parameters[4];
            mu_2 = parameters[1];
        }
        num_iter
    }

    /// `moments` must hold at least `min` entries.
    fn require_moments(&self, min: usize) -> Result<(), InputError> {
        if self.moments.len() < min {
            return Err(InputError::TooShort { name: "moments", len: self.moments.len(), min });
        }
        Ok(())
    }

    /// The fit reads five raw moments and dispatches on `variant`; say so up front instead of
    /// indexing out of bounds or discarding every `fit` error.
    fn validate(&self) -> Result<(), InputError> {
        self.require_moments(5)?;
        if !matches!(self.variant, 1 | 2) {
            return Err(InputError::OutOfRange {
                name: "variant",
                value: self.variant as f64,
                expected: "1 (four moments) or 2 (five moments)",
            });
        }
        Ok(())
    }

    /// One EF3M search: runs [`M2N::fit`] from each start
    /// `mu_2 = m_1 + i * epsilon * factor * sigma`, `i = 1, 2, ...` up to about
    /// `1 / epsilon`, where `sigma` is the standard deviation implied by the moments, and
    /// returns the best fit found.
    ///
    /// `epsilon_override`, when `Some`, replaces [`M2N::epsilon`]. The search state ([`M2N::parameters`],
    /// [`M2N::error`]) is reset first, and holds the best fit afterwards.
    ///
    /// Variant 2 with a non-zero mean is fitted about the mean: the search runs on the
    /// moments of `X - E[X]` and the mean is added back to `mu_1` and `mu_2`, so a negative
    /// `mu_2` can be recovered (#115). The row's `error` is always measured against the raw
    /// moments in [`M2N::moments`].
    ///
    /// Returns at most one row: none if no start produced an admissible iterate that
    /// improves on the initial error (the sum of squared moments). The result is random.
    ///
    /// # Errors
    ///
    /// - [`InputError::TooShort`] if [`M2N::moments`] has fewer than five entries.
    /// - [`InputError::OutOfRange`] if [`M2N::variant`] is not 1 or 2, or if the epsilon in
    ///   use (`epsilon_override`, else [`M2N::epsilon`]) is not finite and `> 0`. Zero used to
    ///   build a start grid of `usize::MAX` points that never finished, and a negative or NaN
    ///   value silently gave an empty result. Nothing is modified when this is returned.
    ///
    /// Non-finite or inconsistent moments (for example a negative implied variance) are not
    /// errors: every attempt fails and the result is empty. The grid has about `1 / epsilon`
    /// points, so the run time grows as `epsilon` shrinks.
    ///
    /// ```
    /// use openquant::ef3m::M2N;
    ///
    /// // Exact moments of 0.7 N(-1, 1) + 0.3 N(2, 0.5^2), fitted with five moments.
    /// let truth = [-1.0, 2.0, 1.0, 0.5, 0.7];
    /// let moments = M2N::with_defaults(vec![]).get_moments(&truth, true).unwrap();
    /// let mut m2n = M2N::new(moments.clone(), 1e-4, 5.0, 1, 2, 100_000, 1);
    /// let rows = m2n.single_fit_loop(None).unwrap();
    /// assert_eq!(rows.len(), 1);
    /// let fit = [rows[0].mu_1, rows[0].mu_2, rows[0].sigma_1, rows[0].sigma_2, rows[0].p_1];
    /// // The starting p_1 is random, but at this epsilon the fit recovers the mixture.
    /// assert!(fit.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 0.05), "{fit:?}");
    /// // The row's error is the squared error of its own parameters on the raw moments.
    /// let implied = m2n.get_moments(&fit, true).unwrap();
    /// let error: f64 = moments.iter().zip(&implied).map(|(a, b)| (a - b).powi(2)).sum();
    /// assert!((error - rows[0].error).abs() < 1e-9);
    ///
    /// // Fewer than five moments are rejected.
    /// assert!(M2N::with_defaults(vec![0.0, 1.0]).single_fit_loop(None).is_err());
    /// ```
    pub fn single_fit_loop(
        &mut self,
        epsilon_override: Option<f64>,
    ) -> Result<Vec<FitResultRow>, InputError> {
        self.validate()?;
        let (name, eps) = match epsilon_override {
            Some(eps) => ("epsilon_override", eps),
            None => ("epsilon", self.epsilon),
        };
        if !(eps.is_finite() && eps > 0.0) {
            return Err(InputError::OutOfRange { name, value: eps, expected: "finite and > 0" });
        }
        self.epsilon = eps;
        if self.variant == 2 && self.moments[0] != 0.0 {
            return self.centred_fit_loop();
        }
        self.parameters = vec![0.0; 5];
        self.error = self.moments.iter().map(|m| m * m).sum();

        let std_dev = centered_moment(&self.moments, 2)?.sqrt();
        let upper = (1.0 / self.epsilon).max(1.0) as usize;
        let mut err_min = self.error;
        let mut best: Option<FitResultRow> = None;

        for i in 1..upper {
            let mu_2_i = i as f64 * self.epsilon * self.factor * std_dev + self.moments[0];
            let _ = self.fit(mu_2_i);
            if self.error < err_min {
                err_min = self.error;
                best = Some(FitResultRow {
                    mu_1: self.parameters[0],
                    mu_2: self.parameters[1],
                    sigma_1: self.parameters[2],
                    sigma_2: self.parameters[3],
                    p_1: self.parameters[4],
                    error: err_min,
                });
            }
        }

        Ok(best.into_iter().collect())
    }

    /// The five-moment variant fitted about the mean, then shifted back.
    ///
    /// `iter_5` solves the fourth-moment equation for mu_2² and, as in López de Prado and
    /// Foreman (2014), takes the positive root. On raw moments that means mu_2 can never be
    /// negative, so a mixture whose upper component has a negative mean cannot be recovered
    /// (#115). Every start of the search puts mu_2 above the mean, so after centring the upper
    /// component's mean is positive and the positive root is the right one, wherever the mixture
    /// sits. The run is otherwise the same algorithm on the moments of X − E[X]; the returned
    /// row's `error` is recomputed against the raw moments, the ones the caller passed.
    fn centred_fit_loop(&mut self) -> Result<Vec<FitResultRow>, InputError> {
        let mean = self.moments[0];
        let mut centred = self.clone();
        centred.moments = (1..=5)
            .map(|order| centered_moment(&self.moments, order))
            .collect::<Result<Vec<f64>, InputError>>()?;
        centred.moments[0] = 0.0;
        let mut rows = centred.single_fit_loop(None)?;

        for row in &mut rows {
            row.mu_1 += mean;
            row.mu_2 += mean;
            row.error = self.moment_error(&[row.mu_1, row.mu_2, row.sigma_1, row.sigma_2, row.p_1]);
        }
        if centred.parameters.iter().all(|p| *p == 0.0) {
            // No start produced an admissible iterate: same state as the uncentred loop leaves.
            self.parameters = vec![0.0; 5];
            self.error = self.moments.iter().map(|m| m * m).sum();
        } else {
            // `fit` only ever stores five parameters.
            let p = &centred.parameters;
            let parameters = [p[0] + mean, p[1] + mean, p[2], p[3], p[4]];
            self.error = self.moment_error(&parameters);
            self.parameters = parameters.to_vec();
        }
        Ok(rows)
    }

    fn moment_error(&mut self, parameters: &[f64; 5]) -> f64 {
        let fitted = self.get_moments(parameters, true).unwrap_or_default();
        self.moments.iter().zip(fitted.iter()).map(|(a, b)| (a - b).powi(2)).sum()
    }

    /// Runs [`M2N::single_fit_loop`] [`M2N::n_runs`] times, each on a fresh copy of `self`,
    /// and concatenates the rows (at most one per run). `self` is not modified.
    ///
    /// The runs execute serially (the deprecated `num_workers` field is ignored). Summarise the
    /// rows with [`most_likely_parameters`].
    ///
    /// # Errors
    ///
    /// Any error of [`M2N::single_fit_loop`]: [`InputError::TooShort`] for fewer than five
    /// moments, [`InputError::OutOfRange`] for a `variant` other than 1 or 2 or an `epsilon`
    /// that is not finite and `> 0`.
    ///
    /// ```
    /// use openquant::ef3m::{most_likely_parameters, M2N};
    ///
    /// let truth = [-1.0, 2.0, 1.0, 0.5, 0.7];
    /// let moments = M2N::with_defaults(vec![]).get_moments(&truth, true).unwrap();
    /// let rows = M2N::new(moments, 1e-4, 5.0, 5, 2, 100_000, 1).mp_fit().unwrap();
    /// assert_eq!(rows.len(), 5);
    /// let fit = most_likely_parameters(&rows, None, 1_000);
    /// assert!((fit["mu_2"] - 2.0).abs() < 0.05 && (fit["p_1"] - 0.7).abs() < 0.05);
    ///
    /// let bad_variant = M2N::new(vec![0.0; 5], 1e-4, 5.0, 1, 3, 100, 1);
    /// assert!(bad_variant.mp_fit().is_err());
    /// ```
    pub fn mp_fit(&self) -> Result<Vec<FitResultRow>, InputError> {
        let mut out = Vec::new();
        for _ in 0..self.n_runs {
            let mut worker = self.clone();
            out.extend(worker.single_fit_loop(None)?);
        }
        Ok(out)
    }
}

/// The `order`-th centred moment `E[(x - m_1)^order]` from raw moments
/// `moments = [E[x], E[x^2], ...]`.
///
/// Uses the binomial expansion `sum_j C(order, j) (-m_1)^j m_{order - j}` with `m_0 = 1`, so
/// only the first `order` raw moments are read. Order 0 gives 1 and order 1 gives 0 (up to
/// rounding).
///
/// # Errors
///
/// [`InputError::TooShort`] if `moments` has fewer than `max(order, 1)` entries.
///
/// ```
/// use openquant::ef3m::centered_moment;
///
/// // N(1, 2^2): E[x] = 1, E[x^2] = 5, E[x^3] = 13.
/// let raw = [1.0, 5.0, 13.0];
/// assert_eq!(centered_moment(&raw, 2).unwrap(), 4.0); // the variance
/// assert_eq!(centered_moment(&raw, 3).unwrap(), 0.0); // symmetric
/// assert!(centered_moment(&raw, 4).is_err());
/// ```
pub fn centered_moment(moments: &[f64], order: usize) -> Result<f64, InputError> {
    // The order-th centred moment is built from the first `order` raw moments.
    if moments.len() < order.max(1) {
        return Err(InputError::TooShort {
            name: "moments",
            len: moments.len(),
            min: order.max(1),
        });
    }
    let mut moment_c = 0.0;
    for j in 0..=order {
        let combin = comb(order, j);
        let a_1 = if j == order { 1.0 } else { moments[order - j - 1] };
        moment_c += (-1.0f64).powi(j as i32) * combin * moments[0].powi(j as i32) * a_1;
    }
    Ok(moment_c)
}

/// Raw moments `[E[x], E[x^2], ...]` from centred moments and the mean.
///
/// `central_moments` starts at the **first** centred moment, which is 0:
/// `[0, E[(x - mu)^2], E[(x - mu)^3], ...]`. The output has the same length (at least 1):
/// entry 0 is `dist_mean` and entry `n - 1` is `sum_k C(n, k) c_k mu^(n - k)` with `c_0 = 1`.
/// The first centred moment is used as given in that sum, so pass 0 there.
///
/// ```
/// use openquant::ef3m::raw_moment;
///
/// // N(1, 2^2): centred moments 0, 4, 0; raw moments 1, 5, 13.
/// assert_eq!(raw_moment(&[0.0, 4.0, 0.0], 1.0), [1.0, 5.0, 13.0]);
/// ```
pub fn raw_moment(central_moments: &[f64], dist_mean: f64) -> Vec<f64> {
    let mut raw_moments = vec![dist_mean];
    let mut central = vec![1.0];
    central.extend_from_slice(central_moments);
    for n_i in 2..central.len() {
        let mut moment_n = 0.0;
        for (k, ck) in central.iter().take(n_i + 1).enumerate() {
            moment_n += comb(n_i, k) * ck * dist_mean.powi((n_i - k) as i32);
        }
        raw_moments.push(moment_n);
    }
    raw_moments
}

/// The most likely value of each parameter over many fits: for each column of `data`
/// separately, the peak of a Gaussian kernel density, rounded to five decimals.
///
/// This is how López de Prado and Foreman (2014) summarise the random EF3M runs. Columns
/// are `mu_1`, `mu_2`, `sigma_1`, `sigma_2`, `p_1` and `error`; those named in
/// `ignore_columns` are skipped (default `["error"]`; `Some(&[])` keeps all six, unknown
/// names are ignored). The bandwidth is `std * n^(-1/5)` (population standard deviation,
/// floored at `1e-6`), and the density is evaluated on `max(res, 10)` evenly spaced points
/// from the column's minimum to its maximum; the first highest point wins. A column whose
/// values are all equal (within `1e-15`) returns that value.
///
/// The returned values may come from different runs and need not reproduce the moments;
/// the runs must also agree on which component is which for the mode to mean anything.
/// Returns an empty map for empty `data`.
///
/// ```
/// use openquant::ef3m::{most_likely_parameters, FitResultRow};
///
/// let row = |mu_2: f64| FitResultRow {
///     mu_1: -1.0, mu_2, sigma_1: 1.0, sigma_2: 0.5, p_1: 0.7, error: 0.0,
/// };
/// // Three runs agree on mu_2 = 2, one outlier at 3.
/// let rows = [row(2.0), row(2.0), row(2.0), row(3.0)];
/// let fit = most_likely_parameters(&rows, None, 11);
/// assert_eq!(fit["mu_2"], 2.0);
/// assert_eq!(fit["p_1"], 0.7); // constant column
/// assert!(!fit.contains_key("error")); // ignored by default
/// ```
pub fn most_likely_parameters(
    data: &[FitResultRow],
    ignore_columns: Option<&[&str]>,
    res: usize,
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if data.is_empty() {
        return out;
    }
    let ignored: HashSet<&str> = ignore_columns.unwrap_or(&["error"]).iter().copied().collect();

    let columns: [(&str, Vec<f64>); 6] = [
        ("mu_1", data.iter().map(|r| r.mu_1).collect()),
        ("mu_2", data.iter().map(|r| r.mu_2).collect()),
        ("sigma_1", data.iter().map(|r| r.sigma_1).collect()),
        ("sigma_2", data.iter().map(|r| r.sigma_2).collect()),
        ("p_1", data.iter().map(|r| r.p_1).collect()),
        ("error", data.iter().map(|r| r.error).collect()),
    ];

    for (name, vals) in columns {
        if ignored.contains(name) {
            continue;
        }
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() < 1e-15 {
            out.insert(name.to_string(), round_to_5(min));
            continue;
        }

        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n.max(1.0);
        let std = var.sqrt().max(1e-12);
        let h = (std * n.powf(-1.0 / 5.0)).max(1e-6);

        let steps = res.max(10);
        let dx = (max - min) / (steps as f64 - 1.0);
        let mut best_x = min;
        let mut best_y = f64::NEG_INFINITY;
        for i in 0..steps {
            let x = min + dx * i as f64;
            let y = vals
                .iter()
                .map(|v| {
                    let u = (x - v) / h;
                    (-0.5 * u * u).exp()
                })
                .sum::<f64>()
                / (n * h * (2.0 * std::f64::consts::PI).sqrt());
            if y > best_y {
                best_y = y;
                best_x = x;
            }
        }
        out.insert(name.to_string(), round_to_5(best_x));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::M2N;

    /// #186 item 25: `fit` ran `max_iter + 1` iterations (and one with `max_iter = 0`).
    #[test]
    fn fit_runs_at_most_max_iter_iterations() {
        let truth = [-1.0, 2.0, 1.0, 0.5, 0.7];
        let moments = M2N::with_defaults(vec![]).get_moments(&truth, true).unwrap();
        for variant in [1, 2] {
            for max_iter in [0, 1, 3] {
                // Started at the truth, a fixed point, every step is admissible; with
                // epsilon = 0 the convergence test never passes, so only max_iter stops it.
                let mut m2n = M2N::new(moments.clone(), 0.0, 5.0, 1, variant, max_iter, 1);
                assert_eq!(m2n.iterate_from(2.0, 0.7), max_iter, "variant {variant}");
            }
        }
    }

    /// #186 item 25: the defaults use variant 2, the one the docs page recommends.
    #[test]
    fn defaults_use_the_five_moment_variant() {
        assert_eq!(M2N::with_defaults(vec![]).variant, 2);
    }
}
