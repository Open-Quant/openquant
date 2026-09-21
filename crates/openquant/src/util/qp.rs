//! A small dense convex QP solver for portfolio problems.
//!
//! Solves `minimise 1/2 x' P x  subject to  l <= A x <= u` with `P` positive semidefinite, by
//! ADMM in the OSQP formulation (Stellato et al., 2020), followed by a polish step that solves
//! the KKT system on the identified active set. There is no linear term: none of the portfolio
//! problems here has one, which also makes the minimiser invariant to the scale of `P`.
//!
//! It is meant for tens of assets, not thousands: every iteration is a dense back-substitution.

use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum QpError {
    /// Shapes of `P`, `A`, `l`, `u` do not agree, or a lower bound exceeds its upper bound.
    Malformed,
    /// The iteration did not reach the tolerance; the constraints are probably infeasible.
    NotConverged,
}

const SIGMA: f64 = 1e-6;
const RHO: f64 = 0.1;
/// OSQP weights equality constraints more heavily; it speeds convergence markedly.
const RHO_EQUALITY_FACTOR: f64 = 1e3;
const RELAXATION: f64 = 1.6;
const MAX_ITERATIONS: usize = 50_000;
const TOLERANCE: f64 = 1e-9;
/// A constraint counts as active in the polish step when it is this close to a bound.
const ACTIVE_TOLERANCE: f64 = 1e-6;

pub(crate) fn solve_qp(
    p: &DMatrix<f64>,
    a: &DMatrix<f64>,
    lower: &[f64],
    upper: &[f64],
) -> Result<Vec<f64>, QpError> {
    let n = p.nrows();
    let m = a.nrows();
    if n == 0
        || p.ncols() != n
        || a.ncols() != n
        || lower.len() != m
        || upper.len() != m
        || lower.iter().zip(upper).any(|(l, u)| l > u || l.is_nan() || u.is_nan())
    {
        return Err(QpError::Malformed);
    }

    // Scale every constraint row to a largest coefficient of one. The feasible set is unchanged,
    // but ADMM converges at a rate set by the conditioning of A: a return constraint with
    // coefficients near 0.05 beside budget and box rows of 1 stalled it until MAX_ITERATIONS,
    // and the failure was reported as an infeasible problem.
    let mut a = a.clone();
    let (mut lower, mut upper) = (lower.to_vec(), upper.to_vec());
    for i in 0..m {
        let largest = a.row(i).iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if largest > 0.0 && largest.is_finite() {
            a.row_mut(i).iter_mut().for_each(|v| *v /= largest);
            lower[i] /= largest;
            upper[i] /= largest;
        }
    }
    let (a, lower, upper) = (&a, lower.as_slice(), upper.as_slice());

    // The minimiser does not depend on the scale of P; bring it to order one so that RHO and
    // SIGMA mean the same thing for daily and for annualised covariances.
    let scale = p.diagonal().iter().map(|v| v.abs()).sum::<f64>() / n as f64;
    let p = if scale > 0.0 { p / scale } else { p.clone() };

    let rho: Vec<f64> = lower
        .iter()
        .zip(upper)
        .map(|(l, u)| if (u - l).abs() < 1e-12 { RHO * RHO_EQUALITY_FACTOR } else { RHO })
        .collect();
    let rho_diag = DMatrix::from_diagonal(&DVector::from_vec(rho.clone()));

    let kkt = &p + DMatrix::identity(n, n) * SIGMA + a.transpose() * &rho_diag * a;
    let factor = kkt.cholesky().ok_or(QpError::Malformed)?;

    let mut x = DVector::zeros(n);
    let mut z = DVector::zeros(m);
    let mut y = DVector::zeros(m);

    let mut converged = false;
    for _ in 0..MAX_ITERATIONS {
        let rhs = &x * SIGMA + a.transpose() * (rho_diag.clone() * &z - &y);
        let x_next = factor.solve(&rhs);
        let ax = a * &x_next;
        let relaxed = &ax * RELAXATION + &z * (1.0 - RELAXATION);

        let mut z_next = DVector::zeros(m);
        for i in 0..m {
            z_next[i] = (relaxed[i] + y[i] / rho[i]).clamp(lower[i], upper[i]);
        }
        for i in 0..m {
            y[i] += rho[i] * (relaxed[i] - z_next[i]);
        }
        x = x_next;
        z = z_next;

        let primal = (&ax - &z).amax();
        let dual = (&p * &x + a.transpose() * &y).amax();
        if primal < TOLERANCE && dual < TOLERANCE {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(QpError::NotConverged);
    }

    Ok(polish(&p, a, lower, upper, &x, &z).unwrap_or(x).iter().copied().collect())
}

/// Solve the equality-constrained problem on the active set found by ADMM. ADMM gets the
/// active set right long before it gets the last digits right; this supplies the digits. The
/// result is used only if it is feasible.
fn polish(
    p: &DMatrix<f64>,
    a: &DMatrix<f64>,
    lower: &[f64],
    upper: &[f64],
    x: &DVector<f64>,
    z: &DVector<f64>,
) -> Option<DVector<f64>> {
    let n = p.nrows();
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    for i in 0..a.nrows() {
        if (z[i] - lower[i]).abs() < ACTIVE_TOLERANCE {
            rows.push(i);
            targets.push(lower[i]);
        } else if (z[i] - upper[i]).abs() < ACTIVE_TOLERANCE {
            rows.push(i);
            targets.push(upper[i]);
        }
    }
    let k = rows.len();
    if k == 0 {
        return None;
    }

    let mut kkt = DMatrix::zeros(n + k, n + k);
    kkt.view_mut((0, 0), (n, n)).copy_from(p);
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..n {
            kkt[(n + r, j)] = a[(i, j)];
            kkt[(j, n + r)] = a[(i, j)];
        }
        // A tiny negative diagonal keeps the system solvable when active rows are dependent.
        kkt[(n + r, n + r)] = -1e-12;
    }
    let mut rhs = DVector::zeros(n + k);
    for (r, target) in targets.iter().enumerate() {
        rhs[n + r] = *target;
    }

    let solution = kkt.lu().solve(&rhs)?;
    let polished = DVector::from_iterator(n, solution.iter().take(n).copied());
    if polished.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let ax = a * &polished;
    let feasible = (0..a.nrows()).all(|i| ax[i] >= lower[i] - 1e-9 && ax[i] <= upper[i] + 1e-9);
    let no_worse = (polished.transpose() * p * &polished)[(0, 0)]
        <= (x.transpose() * p * x)[(0, 0)] * (1.0 + 1e-6) + 1e-15;
    (feasible && no_worse).then_some(polished)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simplex_with_box(n: usize, lo: f64, hi: f64) -> (DMatrix<f64>, Vec<f64>, Vec<f64>) {
        let mut a = DMatrix::zeros(n + 1, n);
        for j in 0..n {
            a[(0, j)] = 1.0;
            a[(j + 1, j)] = 1.0;
        }
        let mut lower = vec![lo; n + 1];
        let mut upper = vec![hi; n + 1];
        lower[0] = 1.0;
        upper[0] = 1.0;
        (a, lower, upper)
    }

    #[test]
    fn uncorrelated_assets_get_inverse_variance_weights() {
        // min w'Cw on the simplex with diagonal C has the closed form w_i ~ 1 / C_ii.
        let variances = [0.04, 0.01, 0.09, 0.0025];
        let p = DMatrix::from_diagonal(&DVector::from_row_slice(&variances));
        let (a, lower, upper) = simplex_with_box(4, 0.0, 1.0);
        let w = solve_qp(&p, &a, &lower, &upper).unwrap();

        let total: f64 = variances.iter().map(|v| 1.0 / v).sum();
        for (got, v) in w.iter().zip(variances) {
            assert!((got - (1.0 / v) / total).abs() < 1e-9, "got {got}");
        }
    }

    #[test]
    fn a_binding_lower_bound_is_honoured_not_clamped_afterwards() {
        // Two perfectly-hedging assets and one poor one. Unconstrained, the minimum-variance
        // portfolio shorts asset 2; long-only, it must sit exactly at zero and the other two
        // must be re-optimised, which is not what clamping the unconstrained answer gives.
        let p = DMatrix::from_row_slice(
            3,
            3,
            &[0.010, 0.002, 0.012, 0.002, 0.020, 0.016, 0.012, 0.016, 0.050],
        );
        let (a, lower, upper) = simplex_with_box(3, 0.0, 1.0);
        let w = solve_qp(&p, &a, &lower, &upper).unwrap();

        assert!(w[2].abs() < 1e-9, "asset 2 should be at its bound, got {}", w[2]);
        // With w2 = 0 the problem is the two-asset closed form.
        let w0 = (0.020 - 0.002) / (0.010 + 0.020 - 2.0 * 0.002);
        assert!((w[0] - w0).abs() < 1e-9 && (w[1] - (1.0 - w0)).abs() < 1e-9, "got {w:?}");
    }

    #[test]
    fn infeasible_constraints_are_reported() {
        let p = DMatrix::identity(2, 2);
        let (a, lower, upper) = simplex_with_box(2, 0.0, 0.3); // at most 0.6 in total
        assert_eq!(solve_qp(&p, &a, &lower, &upper), Err(QpError::NotConverged));
    }
}
