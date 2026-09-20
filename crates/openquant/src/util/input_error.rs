//! The error returned by functions whose only failure mode is an invalid argument.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum InputError {
    /// `name` must contain at least `min` values.
    TooShort { name: &'static str, len: usize, min: usize },
    /// `name` must be as long as the series it is paired with.
    LengthMismatch { name: &'static str, len: usize, expected: usize },
    /// `name` holds a value outside the range the function is defined on.
    OutOfRange { name: &'static str, value: f64, expected: &'static str },
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
