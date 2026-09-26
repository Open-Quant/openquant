//! The error returned by functions whose only failure mode is an invalid argument.
//!
//! [`InputError`] is shared by modules that validate their arguments but have no richer error
//! type of their own (for example [`crate::util::fast_ewma`], [`crate::sampling`],
//! [`crate::ef3m`] and [`crate::backtest_statistics`]). Each variant names the offending
//! argument, so the [`Display`](std::fmt::Display) message can be shown to a user as is.
//!
//! ```
//! use openquant::util::fast_ewma::ewma;
//! use openquant::util::InputError;
//!
//! let err = ewma(&[1.0, 2.0], 0).unwrap_err();
//! assert_eq!(
//!     err,
//!     InputError::OutOfRange { name: "window", value: 0.0, expected: "a positive integer" }
//! );
//! assert_eq!(err.to_string(), "'window' is 0, expected a positive integer");
//! ```

use std::fmt;

/// An argument was rejected before any computation ran.
///
/// `name` is the argument (or, for rows of a matrix, a short description of it) as it appears
/// in the function signature.
#[derive(Debug, Clone, PartialEq)]
pub enum InputError {
    /// `name` must contain at least `min` values.
    TooShort {
        /// The argument that was too short.
        name: &'static str,
        /// Its actual length.
        len: usize,
        /// The minimum length the function needs.
        min: usize,
    },
    /// `name` must be as long as the series it is paired with.
    LengthMismatch {
        /// The argument whose length is wrong.
        name: &'static str,
        /// Its actual length.
        len: usize,
        /// The length it must have to match its partner.
        expected: usize,
    },
    /// `name` holds a value outside the range the function is defined on.
    OutOfRange {
        /// The argument that is out of range.
        name: &'static str,
        /// The offending value (converted to `f64` for integer arguments).
        value: f64,
        /// A human-readable description of the accepted range.
        expected: &'static str,
    },
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputError::TooShort { name, len, min } => {
                write!(f, "'{name}' has length {len}, need at least {min}")
            }
            InputError::LengthMismatch { name, len, expected } => {
                write!(f, "'{name}' has length {len}, expected {expected}")
            }
            InputError::OutOfRange { name, value, expected } => {
                write!(f, "'{name}' is {value}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for InputError {}

pub(crate) fn same_length(
    name: &'static str,
    values: &[f64],
    expected: usize,
) -> Result<(), InputError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(InputError::LengthMismatch { name, len: values.len(), expected })
    }
}
