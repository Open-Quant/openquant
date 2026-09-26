//! Small building blocks shared across modules: the EWMA primitive, volatility estimators and
//! the common invalid-argument error.
//!
//! - [`fast_ewma`]: span-style exponentially weighted moving average (the decay convention every
//!   EWMA-derived series in the crate shares).
//! - [`volatility`]: daily volatility (AFML Snippet 3.1) and range-based estimators.
//! - [`input_error`]: [`InputError`], returned by functions whose only failure mode is a bad
//!   argument.

// `#![deny(missing_docs)]` is set per submodule, not here, because a lint level set in this
// file would also apply to `volatility`, which is not fully documented yet.

pub mod fast_ewma;
pub mod input_error;
pub(crate) mod linkage;
pub(crate) mod qp;
pub(crate) mod resample;
/// Volatility estimators used by labeling and risk code (AFML Snippet 3.1 and range-based
/// estimators).
pub mod volatility;

pub use input_error::InputError;
