//! Exact finite binary64 sums/products for exceptional numerical recovery.
//! Ordinary numerical paths do not use this integer arithmetic.
use std::cmp::Ordering;
#[derive(Clone, Debug, Default)]
struct Big {
    words: Vec<u64>,
}
impl Big {
    fn trim(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }
    fn bits(&self) -> usize {
        self.words
            .last()
            .map_or(0, |v| 64 * (self.words.len() - 1) + (64 - v.leading_zeros()) as usize)
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.words
            .len()
            .cmp(&other.words.len())
            .then_with(|| self.words.iter().rev().cmp(other.words.iter().rev()))
    }
    fn add_word(&mut self, mut index: usize, mut value: u64) {
        while value != 0 {
            if self.words.len() <= index {
                self.words.resize(index + 1, 0);
            }
            let (next, carry) = self.words[index].overflowing_add(value);
            self.words[index] = next;
            value = u64::from(carry);
            index += 1;
        }
    }
    fn add_product(&mut self, product: u128, shift: usize) {
        let index = shift / 64;
        let offset = shift % 64;
        let low = product as u64;
        let high = (product >> 64) as u64;
        self.add_word(index, low << offset);
        self.add_word(index + 1, high << offset);
        if offset != 0 {
            self.add_word(index + 1, low >> (64 - offset));
            self.add_word(index + 2, high >> (64 - offset));
        }
    }
    fn shifted(&self, shift: usize) -> Self {
        if self.words.is_empty() {
            return Self::default();
        }
        let offset = shift % 64;
        let whole = shift / 64;
        let mut result = Self { words: vec![0; self.words.len() + whole + 1] };
        for (index, &word) in self.words.iter().enumerate() {
            result.words[index + whole] |= word << offset;
            if offset != 0 {
                result.words[index + whole + 1] |= word >> (64 - offset);
            }
        }
        result.trim();
        result
    }
    fn subtract(&mut self, other: &Self) {
        assert!(self.compare(other) != Ordering::Less);
        let mut borrow = false;
        for index in 0..self.words.len() {
            let (first, b1) =
                self.words[index].overflowing_sub(other.words.get(index).copied().unwrap_or(0));
            let (next, b2) = first.overflowing_sub(u64::from(borrow));
            self.words[index] = next;
            borrow = b1 || b2;
        }
        assert!(!borrow);
        self.trim();
    }
    fn small_division(mut self, denominator: &Self) -> (u64, Self) {
        if self.compare(denominator) == Ordering::Less {
            return (0, self);
        }
        let shift = self.bits() - denominator.bits();
        assert!(shift < 64);
        let mut quotient = 0;
        for bit in (0..=shift).rev() {
            let subtrahend = denominator.shifted(bit);
            if self.compare(&subtrahend) != Ordering::Less {
                self.subtract(&subtrahend);
                quotient |= 1_u64 << bit;
            }
        }
        (quotient, self)
    }
}
fn integer_parts(value: f64) -> (u64, i32) {
    let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1_u64 << 52), exponent - 1075)
    }
}
fn exact_product_sum(left: &[f64], right: &[f64]) -> Big {
    let mut total = Big::default();
    for (&x, &y) in left.iter().zip(right) {
        let (mx, ex) = integer_parts(x);
        let (my, ey) = integer_parts(y);
        total.add_product(mx as u128 * my as u128, (ex + ey + 2148) as usize);
    }
    total
}
fn exact_nonnegative_sum(values: &[f64]) -> Big {
    let mut total = Big::default();
    for &value in values {
        let (mantissa, exponent) = integer_parts(value);
        // Unlike product sums, these integers share a 2^-1074 unit.
        total.add_product(mantissa as u128, (exponent + 1074) as usize);
    }
    total
}
fn rounded_ratio(numerator: &Big, denominator: &Big) -> f64 {
    if denominator.bits() == 0 {
        return f64::NAN;
    }
    if numerator.bits() == 0 {
        return 0.0;
    }
    let mut exponent = numerator.bits() as i32 - denominator.bits() as i32;
    let compare = if exponent >= 0 {
        numerator.compare(&denominator.shifted(exponent as usize))
    } else {
        numerator.shifted((-exponent) as usize).compare(denominator)
    };
    if compare == Ordering::Less {
        exponent -= 1;
    }
    if exponent > 1023 {
        return f64::INFINITY;
    }
    if exponent < -1075 {
        return 0.0;
    }
    let subnormal = exponent < -1022;
    let shift = if subnormal { 1074 } else { 52 - exponent };
    let (scaled_num, scaled_den) = if shift >= 0 {
        (numerator.shifted(shift as usize), denominator.clone())
    } else {
        (numerator.clone(), denominator.shifted((-shift) as usize))
    };
    let (mut significand, remainder) = scaled_num.small_division(&scaled_den);
    match remainder.shifted(1).compare(&scaled_den) {
        Ordering::Greater => significand += 1,
        Ordering::Equal if significand & 1 != 0 => significand += 1,
        _ => {}
    }
    if subnormal {
        return f64::from_bits(significand);
    }
    if significand == 1_u64 << 53 {
        significand >>= 1;
        exponent += 1;
    }
    if exponent > 1023 {
        return f64::INFINITY;
    }
    f64::from_bits((((exponent + 1023) as u64) << 52) | (significand - (1_u64 << 52)))
}
pub(crate) fn nonnegative_origin_slope(returns: &[f64], volume: &[f64]) -> f64 {
    if returns.len() != volume.len()
        || returns.iter().any(|v| !v.is_finite())
        || volume.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return f64::NAN;
    }
    rounded_ratio(&exact_product_sum(volume, returns), &exact_product_sum(volume, volume))
}

pub(crate) fn nonnegative_sum_ratio(numerator: &[f64], denominator: &[f64]) -> f64 {
    if numerator.iter().chain(denominator).any(|value| !value.is_finite() || *value < 0.0) {
        return f64::NAN;
    }
    let num = exact_nonnegative_sum(numerator);
    let den = exact_nonnegative_sum(denominator);
    if num.bits() == 0 && den.bits() != 0 {
        // Every numerator is an actual signed zero, not an underflowed product.
        return numerator.iter().sum();
    }
    rounded_ratio(&num, &den)
}

pub(crate) fn signed_origin_slope(response: &[f64], regressor: &[f64]) -> f64 {
    if response.len() != regressor.len()
        || response.iter().chain(regressor).any(|value| !value.is_finite())
    {
        return f64::NAN;
    }
    let mut positive = Big::default();
    let mut negative = Big::default();
    let mut denominator = Big::default();
    let mut zero_sum = -0.0;
    for (&x, &y) in regressor.iter().zip(response) {
        let (mx, ex) = integer_parts(x);
        let (my, ey) = integer_parts(y);
        denominator.add_product(mx as u128 * mx as u128, (2 * ex + 2148) as usize);
        if mx == 0 || my == 0 {
            // Here the true product is zero, not a rounded underflow. Preserve
            // the legacy sum's sign if every exact product is a signed zero.
            zero_sum += x * y;
        } else {
            let accumulator = if x.is_sign_negative() != y.is_sign_negative() {
                &mut negative
            } else {
                &mut positive
            };
            accumulator.add_product(mx as u128 * my as u128, (ex + ey + 2148) as usize);
        }
    }
    if denominator.bits() == 0 {
        return f64::NAN;
    }
    if positive.bits() == 0 && negative.bits() == 0 {
        return zero_sum;
    }
    if positive.compare(&negative) == Ordering::Less {
        negative.subtract(&positive);
        -rounded_ratio(&negative, &denominator)
    } else {
        positive.subtract(&negative);
        rounded_ratio(&positive, &denominator)
    }
}

/// Correctly round `lower + fraction * (upper - lower)` for finite operands.
/// Called only for overflowing finite quantile spans, with `fraction` in `[0, 1]`.
pub(crate) fn finite_interpolation(lower: f64, upper: f64, fraction: f64) -> f64 {
    let mut positive = Big::default();
    let mut negative = Big::default();
    // Keep all three products exact, including the subtraction implicit in
    // (1 - fraction). Rounding that coefficient first can erase cancellation.
    for (x, y) in [(lower, 1.0), (upper, fraction), (-lower, fraction)] {
        let (mx, ex) = integer_parts(x);
        let (my, ey) = integer_parts(y);
        let sum = if x.is_sign_negative() != y.is_sign_negative() {
            &mut negative
        } else {
            &mut positive
        };
        sum.add_product(mx as u128 * my as u128, (ex + ey + 2148) as usize);
    }
    let mut denominator = Big::default();
    denominator.add_product(1, 2148);
    if positive.compare(&negative) == Ordering::Less {
        negative.subtract(&positive);
        -rounded_ratio(&negative, &denominator)
    } else {
        positive.subtract(&negative);
        rounded_ratio(&positive, &denominator)
    }
}

/// Correctly round `side * (end / start - 1)` for finite positive start,
/// finite nonnegative end and finite side, without rounding intermediate products.
pub(crate) fn scaled_return(end: f64, start: f64, side: f64) -> Option<f64> {
    if ![end, start, side].iter().all(|v| v.is_finite()) || start <= 0.0 || end < 0.0 {
        return None;
    }
    if side == 0.0 {
        return Some(if end < start { -side } else { side });
    }
    if end == start {
        return Some(0.0 * side);
    }
    let (me, ee) = integer_parts(end);
    let (mb, eb) = integer_parts(start);
    let (ms, es) = integer_parts(side);
    let mut end_product = Big::default();
    let mut start_product = Big::default();
    let mut denominator = Big::default();
    // Cancel a common power of two before accumulation. This keeps nearby
    // ordinary-price operands compact without rounding or losing small terms.
    let base = (ee + es).min(eb + es).min(eb);
    end_product.add_product(me as u128 * ms as u128, (ee + es - base) as usize);
    start_product.add_product(mb as u128 * ms as u128, (eb + es - base) as usize);
    denominator.add_product(mb as u128, (eb - base) as usize);
    let (magnitude, negative) = if end < start {
        start_product.subtract(&end_product);
        (rounded_ratio(&start_product, &denominator), !side.is_sign_negative())
    } else {
        end_product.subtract(&start_product);
        (rounded_ratio(&end_product, &denominator), side.is_sign_negative())
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// Compare `side * (price / start - 1)` with `multiple * target` exactly
/// for finite operands with positive start/target and nonnegative price.
/// Multiplying by positive start avoids division; three-factor products use two
/// u128 partial products so their up-to-159-bit significands are never truncated.
pub(crate) fn compare_scaled_return(
    price: f64,
    start: f64,
    side: f64,
    multiple: f64,
    target: f64,
) -> Option<Ordering> {
    if ![price, start, side, multiple, target].iter().all(|v| v.is_finite())
        || start <= 0.0
        || target <= 0.0
        || price < 0.0
    {
        return None;
    }
    let mut positive = Big::default();
    let mut negative = Big::default();
    // All terms share the exponent grid 2^-3222 (three minimum subnormal factors).
    for (x, y, subtract) in [(side, price, false), (side, start, true)] {
        let (mx, ex) = integer_parts(x);
        let (my, ey) = integer_parts(y);
        let accumulator = if x.is_sign_negative() ^ y.is_sign_negative() ^ subtract {
            &mut negative
        } else {
            &mut positive
        };
        accumulator.add_product(mx as u128 * my as u128, (ex + ey + 3222) as usize);
    }
    let (mm, em) = integer_parts(multiple);
    let (mt, et) = integer_parts(target);
    let (ms, es) = integer_parts(start);
    let product = mm as u128 * mt as u128;
    let shift = (em + et + es + 3222) as usize;
    let accumulator =
        if !(multiple.is_sign_negative() ^ target.is_sign_negative() ^ start.is_sign_negative()) {
            &mut negative
        } else {
            &mut positive
        };
    accumulator.add_product((product as u64) as u128 * ms as u128, shift);
    accumulator.add_product((product >> 64) * ms as u128, shift + 64);
    Some(positive.compare(&negative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: u64) -> Big {
        let mut out = Big::default();
        out.add_word(0, value);
        out
    }

    #[test]
    fn carries_shifts_and_borrows_cross_word_boundaries() {
        let mut value = Big { words: vec![u64::MAX; 3] };
        value.add_word(0, 1);
        assert_eq!(value.words, vec![0, 0, 0, 1]);
        assert_eq!(value.shifted(65).words, vec![0, 0, 0, 0, 2]);
        value.subtract(&integer(1));
        assert_eq!(value.words, vec![u64::MAX; 3]);
    }

    #[test]
    fn full_width_products_straddle_words() {
        let mut value = Big::default();
        value.add_product(u128::MAX, 63);
        assert_eq!(value.words, vec![1 << 63, u64::MAX, (1 << 63) - 1]);
        value.add_product(u128::MAX, 63);
        assert_eq!(value.words, vec![0, u64::MAX, u64::MAX]);
    }

    #[test]
    fn division_retains_the_exact_remainder() {
        let mut denominator = integer(1).shifted(80);
        denominator.add_word(0, 3);
        let mut numerator = denominator.shifted(53);
        numerator.subtract(&denominator);
        numerator.add_word(0, 5);
        let (quotient, remainder) = numerator.small_division(&denominator);
        assert_eq!(quotient, (1 << 53) - 1);
        assert_eq!(remainder.words, vec![5]);
    }

    #[test]
    fn rounding_is_ties_to_even_at_normal_and_subnormal_boundaries() {
        let denominator = integer(1).shifted(53);
        assert_eq!(rounded_ratio(&integer((1 << 53) + 1), &denominator), 1.0);
        assert_eq!(
            rounded_ratio(&integer((1 << 53) + 3), &denominator).to_bits(),
            1.0_f64.to_bits() + 2
        );
        let denominator = integer(1).shifted(1075);
        assert_eq!(rounded_ratio(&integer(1), &denominator), 0.0);
        assert_eq!(rounded_ratio(&integer(3), &denominator), f64::from_bits(2));
        assert_eq!(rounded_ratio(&integer((1 << 53) - 1), &denominator), f64::MIN_POSITIVE);
        assert!(rounded_ratio(&Big::default(), &Big::default()).is_nan());
    }

    #[test]
    fn maximum_finite_overflow_midpoint_uses_the_exact_remainder() {
        let threshold = integer((1 << 54) - 1).shifted(970);
        assert_eq!(rounded_ratio(&threshold, &integer(1)), f64::INFINITY);
        let mut below = threshold.clone();
        below.subtract(&integer(1));
        assert_eq!(rounded_ratio(&below, &integer(1)), f64::MAX);
    }

    #[test]
    fn signed_sum_distinguishes_zero_products_from_nonzero_cancellation() {
        assert_eq!(signed_origin_slope(&[-0.0; 2], &[1.0; 2]).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(signed_origin_slope(&[0.0, -0.0], &[1.0; 2]).to_bits(), 0.0_f64.to_bits());
        assert_eq!(signed_origin_slope(&[-0.0; 2], &[-1.0; 2]).to_bits(), 0.0_f64.to_bits());
        assert_eq!(signed_origin_slope(&[1.0, -1.0], &[1.0; 2]).to_bits(), 0.0_f64.to_bits());
        assert_eq!(signed_origin_slope(&[1e16, 3.0, -1e16], &[1.0; 3]), 1.0);
        assert!(signed_origin_slope(&[], &[]).is_nan());
    }

    #[test]
    fn signed_ratio_preserves_negative_underflow_and_ties() {
        assert_eq!(
            signed_origin_slope(&[-f64::from_bits(1)], &[2.0]).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            signed_origin_slope(&[-f64::from_bits(3)], &[2.0]).to_bits(),
            (-f64::from_bits(2)).to_bits()
        );
    }

    #[test]
    fn barrier_comparison_preserves_strict_equality_and_adjacent_prices() {
        assert_eq!(compare_scaled_return(2.0, 1.0, 1.0, 1.0, 1.0), Some(Ordering::Equal));
        assert_eq!(
            compare_scaled_return(2.0_f64.next_up(), 1.0, 1.0, 1.0, 1.0),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compare_scaled_return(2.0_f64.next_down(), 1.0, 1.0, 1.0, 1.0),
            Some(Ordering::Less)
        );
        assert_eq!(compare_scaled_return(2.0, 1.0, -1.0, -1.0, 1.0), Some(Ordering::Equal));
        assert_eq!(
            compare_scaled_return(2.0_f64.next_up(), 1.0, -1.0, -1.0, 1.0),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn barrier_comparison_keeps_low_bits_of_three_factor_products() {
        // Both leading quantities overflow in binary64, but the exact remainder
        // is negative for MAX*MAX and positive after one downward threshold ULP.
        assert_eq!(
            compare_scaled_return(f64::MAX, 1.0, f64::MAX, f64::MAX, f64::MAX),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_scaled_return(f64::MAX, 1.0, f64::MAX, f64::MAX, f64::MAX.next_down()),
            Some(Ordering::Greater)
        );
        let tiny = f64::from_bits(1);
        assert_eq!(compare_scaled_return(1.0, tiny, tiny, 1.0, 1.0), Some(Ordering::Less));
    }

    #[test]
    fn barrier_comparison_rejects_inputs_outside_the_recovery_domain() {
        assert_eq!(compare_scaled_return(-1.0, 1.0, 1.0, 1.0, 1.0), None);
        assert_eq!(compare_scaled_return(1.0, 0.0, 1.0, 1.0, 1.0), None);
        assert_eq!(compare_scaled_return(1.0, 1.0, 1.0, 1.0, 0.0), None);
        assert_eq!(compare_scaled_return(1.0, 1.0, f64::NAN, 1.0, 1.0), None);
        assert_eq!(compare_scaled_return(1.0, 1.0, 1.0, f64::INFINITY, 1.0), None);
    }

    #[test]
    fn scaled_return_cancels_common_exponents_without_rounding() {
        for exponent in [-900, 0, 900] {
            let start = 2.0_f64.powi(exponent);
            assert_eq!(scaled_return(start.next_up(), start, 1.0), Some(f64::EPSILON));
            assert_eq!(scaled_return(start.next_up(), start, -1.0), Some(-f64::EPSILON));
        }
        assert_eq!(
            scaled_return(f64::MAX, f64::MAX.next_down(), 1.0),
            Some(f64::from_bits(0x3ca0000000000001))
        );
    }

    #[test]
    fn scaled_return_preserves_zero_signs_and_rejects_invalid_domain() {
        assert_eq!(scaled_return(3.0, 1e-308, -0.0).unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(scaled_return(1.0, 2.0, 0.0).unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(scaled_return(1.0, 1.0, -1.0).unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(scaled_return(-1.0, 1.0, 1.0), None);
        assert_eq!(scaled_return(1.0, 0.0, 1.0), None);
        assert_eq!(scaled_return(1.0, 1.0, f64::NAN), None);
    }
}

/// Recover an infinite ordinary mean only when every observation is finite.
/// Existing finite, NaN, empty and nonfinite-input results pass through unchanged.
/// The signed sum is exact in units of 2^-1074; division by the exact integer count
/// uses the existing nearest/ties-to-even rational conversion. This is a cold path,
/// not a change to the ordinary shared statistics estimator.
pub(crate) fn recover_overflowed_mean(values: &[f64], ordinary_mean: f64) -> f64 {
    if !ordinary_mean.is_infinite() || values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return ordinary_mean;
    }
    let mut positive = Big::default();
    let mut negative = Big::default();
    for &value in values {
        let (mantissa, exponent) = integer_parts(value);
        let total = if value.is_sign_negative() { &mut negative } else { &mut positive };
        total.add_product(mantissa as u128, (exponent + 1074) as usize);
    }
    let sign = positive.compare(&negative);
    let numerator = match sign {
        Ordering::Greater => {
            positive.subtract(&negative);
            positive
        }
        Ordering::Less => {
            negative.subtract(&positive);
            negative
        }
        Ordering::Equal => return 0.0,
    };
    // numerator is in minimum-subnormal units, so divide by count * 2^1074.
    let mut denominator = Big::default();
    denominator.add_product(values.len() as u128, 1074);
    let result = rounded_ratio(&numerator, &denominator);
    if sign == Ordering::Less {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod mean_recovery_tests {
    use super::recover_overflowed_mean;

    #[test]
    fn overflow_recovery_retains_tiny_cancellation_residuals() {
        let m = f64::MAX;
        let values = [m, m, -m, -m, 1e-308];
        assert_eq!(recover_overflowed_mean(&values, f64::INFINITY), 1e-308 / 5.0);
        assert_eq!(recover_overflowed_mean(&[m, m, -m, -m], f64::INFINITY), 0.0);
    }

    #[test]
    fn subnormal_means_round_ties_to_even_and_keep_negative_underflow_sign() {
        let m = f64::MAX;
        let u = f64::from_bits(1);
        for (a, b, expected) in [(u, 2.0 * u, 0.0), (4.0 * u, 5.0 * u, 2.0 * u)] {
            assert_eq!(
                recover_overflowed_mean(&[m, m, -m, -m, a, b], f64::INFINITY).to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(
            recover_overflowed_mean(&[m, m, -m, -m, -u], f64::INFINITY).to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn ordinary_and_nonfinite_input_results_are_not_reinterpreted() {
        assert_eq!(recover_overflowed_mean(&[1.0, 2.0], -0.0).to_bits(), (-0.0_f64).to_bits());
        assert!(recover_overflowed_mean(&[], f64::NAN).is_nan());
        assert_eq!(recover_overflowed_mean(&[f64::INFINITY], f64::INFINITY), f64::INFINITY);
        assert_eq!(recover_overflowed_mean(&[f64::NAN], f64::NEG_INFINITY), f64::NEG_INFINITY);
    }
}
