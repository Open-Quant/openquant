//! Financial data structures (AFML chapter 2): time, tick, volume, dollar, run and
//! imbalance bars built from a stream of trades.
//!
//! | Builder | A bar closes when | AFML |
//! | --- | --- | --- |
//! | [`time_bars`] | the first trade at least `interval` after the bar's first trade | §2.3.1.1 |
//! | [`standard_bars`] with [`StandardBarType::Tick`] | `threshold` trades have accumulated | §2.3.1.2 |
//! | [`standard_bars`] with [`StandardBarType::Volume`] | cumulative volume reaches `threshold` | §2.3.1.3 |
//! | [`standard_bars`] with [`StandardBarType::Dollar`] | cumulative `price * volume` reaches `threshold` | §2.3.1.4 |
//! | [`imbalance_bars`] | the absolute signed imbalance reaches `threshold` | §2.3.2.1–2, simplified |
//! | [`run_bars`] | `threshold` consecutive same-direction ticks | §2.3.2.3, simplified |
//!
//! Conventions common to all builders:
//!
//! - Trades are in increasing time order; the input is not sorted or validated.
//! - The trade that crosses the threshold belongs to the bar it closes, so bars overshoot.
//! - A trailing partial bar is dropped, except by [`time_bars`], which emits it.
//! - Time bars are anchored to each bar's first trade, not to the wall clock.
//! - Run and imbalance bars use a **constant** threshold, not AFML's adaptive expected
//!   threshold; treat them as fixed-threshold variants.
//! - Trade direction is the tick rule: up-tick `+1`, down-tick `-1`, unchanged price keeps
//!   the previous sign.
//!
//! ```
//! use chrono::{Duration, NaiveDate};
//! use openquant::data_structures::{standard_bars, time_bars, StandardBarType, Trade};
//!
//! # fn main() -> Result<(), openquant::util::InputError> {
//! let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 30, 0).unwrap();
//! let trades: Vec<Trade> = (0..1_000)
//!     .map(|i| Trade {
//!         timestamp: open + Duration::seconds(i * 3),
//!         price: 100.0 + (i as f64 * 0.01).sin(),
//!         volume: 1.0 + (i % 5) as f64,
//!     })
//!     .collect();
//!
//! let dollar = standard_bars(&trades, 25_000.0, StandardBarType::Dollar)?;
//! assert!(dollar[0].dollar_value >= 25_000.0);
//! assert!(dollar[0].start_timestamp <= dollar[0].timestamp);
//!
//! // Nine full five-minute bars and the partial one left over.
//! let five_minute = time_bars(&trades, Duration::minutes(5))?;
//! assert_eq!(five_minute.len(), 10);
//! # Ok(())
//! # }
//! ```

use crate::util::InputError;
use chrono::{Duration, NaiveDateTime};

/// Single trade input.
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
    /// Execution time.
    pub timestamp: NaiveDateTime,
    /// Execution price.
    pub price: f64,
    /// Traded size (shares or contracts).
    pub volume: f64,
}

/// Supported standard bar accumulation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardBarType {
    /// Close after `threshold` trades.
    Tick,
    /// Close when cumulative volume reaches `threshold`.
    Volume,
    /// Close when cumulative `price * volume` reaches `threshold`.
    Dollar,
}

/// Bar output with OHLCV-like fields.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardBar {
    /// Timestamp of the bar's first trade.
    pub start_timestamp: NaiveDateTime,
    /// Timestamp of the bar's last trade (when the bar is known).
    pub timestamp: NaiveDateTime,
    /// Price of the first trade.
    pub open: f64,
    /// Highest trade price.
    pub high: f64,
    /// Lowest trade price.
    pub low: f64,
    /// Price of the last trade.
    pub close: f64,
    /// Total traded volume.
    pub volume: f64,
    /// Total `price * volume`.
    pub dollar_value: f64,
    /// Number of trades in the bar.
    pub tick_count: usize,
}

/// Signed-imbalance accumulation modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImbalanceBarType {
    /// Each signed trade counts 1.
    Tick,
    /// Each signed trade counts its volume.
    Volume,
    /// Each signed trade counts its `price * volume`.
    Dollar,
}

/// Construct standard bars (tick/volume/dollar) from a stream of trades using a static threshold.
///
/// This mirrors the mlfinlab behavior of emitting a bar whenever the chosen metric
/// crosses the threshold and starting accumulation fresh afterward. Any trailing
/// partial bar that does not satisfy the threshold is dropped. An empty slice yields no bars.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `threshold` is not a positive number (zero, negative or
/// `NaN`).
pub fn standard_bars(
    trades: &[Trade],
    threshold: f64,
    bar_type: StandardBarType,
) -> Result<Vec<StandardBar>, InputError> {
    if threshold.is_nan() || threshold <= 0.0 {
        return Err(InputError::OutOfRange {
            name: "threshold",
            value: threshold,
            expected: "a positive number",
        });
    }

    let mut bars = Vec::new();
    if trades.is_empty() {
        return Ok(bars);
    }

    let mut start_idx = 0;
    let mut tick_count = 0usize;
    let mut volume = 0.0;
    let mut dollar_value = 0.0;

    for (i, trade) in trades.iter().enumerate() {
        tick_count += 1;
        volume += trade.volume;
        dollar_value += trade.price * trade.volume;

        let reached = match bar_type {
            StandardBarType::Tick => (tick_count as f64) >= threshold,
            StandardBarType::Volume => volume >= threshold,
            StandardBarType::Dollar => dollar_value >= threshold,
        };

        if reached {
            bars.push(build_bar(&trades[start_idx..=i]));
            start_idx = i + 1;
            tick_count = 0;
            volume = 0.0;
            dollar_value = 0.0;
        }
    }

    Ok(bars)
}

/// Construct time bars using a fixed interval. The interval applies from the start
/// timestamp of the current bar; the trade that crosses the interval boundary is
/// included in the closing bar, and accumulation restarts afterward. Any trailing partial bar
/// is emitted as a final, shorter bar. An empty slice yields no bars.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `interval` is zero or negative.
pub fn time_bars(trades: &[Trade], interval: Duration) -> Result<Vec<StandardBar>, InputError> {
    if interval <= Duration::zero() {
        return Err(InputError::OutOfRange {
            name: "interval",
            value: interval.num_milliseconds() as f64 / 1e3,
            expected: "a positive duration (seconds)",
        });
    }

    let mut bars = Vec::new();
    if trades.is_empty() {
        return Ok(bars);
    }

    let mut start_idx = 0;
    let mut bar_start = trades[0].timestamp;

    for (i, trade) in trades.iter().enumerate() {
        let elapsed = trade.timestamp - bar_start;
        if elapsed >= interval {
            bars.push(build_bar(&trades[start_idx..=i]));
            start_idx = i + 1;
            if start_idx < trades.len() {
                bar_start = trades[start_idx].timestamp;
            }
        }
    }

    if start_idx < trades.len() {
        bars.push(build_bar(&trades[start_idx..]));
    }

    Ok(bars)
}

/// Construct run bars by counting consecutive price-direction runs. A bar closes when
/// `threshold` consecutive moves occur in the same direction. Trailing partial bars
/// that have not met the threshold are dropped. Direction follows the tick rule, so a trade at
/// an unchanged price extends the current run. Fewer than two trades yield no bars.
///
/// This is a fixed-threshold simplification of AFML §2.3.2.3, which compares the count of
/// the dominant side within the bar with an adaptive expectation.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `threshold` is zero.
pub fn run_bars(trades: &[Trade], threshold: usize) -> Result<Vec<StandardBar>, InputError> {
    if threshold == 0 {
        return Err(InputError::OutOfRange {
            name: "threshold",
            value: 0.0,
            expected: "a positive integer",
        });
    }
    if trades.len() < 2 {
        return Ok(Vec::new());
    }

    let mut bars = Vec::new();
    let mut start_idx = 0usize;
    let mut prev_price = trades[0].price;
    let mut prev_sign = 0i8;
    let mut run_len = 0usize;

    for (i, trade) in trades.iter().enumerate().skip(1) {
        let sign = trade_sign(trade.price, prev_price, prev_sign);
        if sign != 0 {
            if sign == prev_sign {
                run_len += 1;
            } else {
                run_len = 1;
                prev_sign = sign;
            }
        }
        prev_price = trade.price;

        if run_len >= threshold {
            bars.push(build_bar(&trades[start_idx..=i]));
            start_idx = i + 1;
            run_len = 0;
            prev_sign = 0;
            if start_idx < trades.len() {
                prev_price = trades[start_idx].price;
            }
        }
    }

    Ok(bars)
}

/// Construct imbalance bars by accumulating signed imbalance (tick, volume, or dollar)
/// until the absolute imbalance crosses `threshold`. Trailing partial bars that have
/// not met the threshold are dropped. Fewer than two trades yield no bars.
///
/// Each trade after the first contributes its tick-rule sign times a weight (1, volume or
/// dollar value, per `bar_type`). This is a fixed-threshold simplification of AFML
/// §2.3.2.1–2, whose threshold is an exponentially weighted expectation.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if `threshold` is not a positive number (zero, negative or
/// `NaN`).
pub fn imbalance_bars(
    trades: &[Trade],
    threshold: f64,
    bar_type: ImbalanceBarType,
) -> Result<Vec<StandardBar>, InputError> {
    if threshold.is_nan() || threshold <= 0.0 {
        return Err(InputError::OutOfRange {
            name: "threshold",
            value: threshold,
            expected: "a positive number",
        });
    }
    if trades.len() < 2 {
        return Ok(Vec::new());
    }

    let mut bars = Vec::new();
    let mut start_idx = 0usize;
    let mut prev_price = trades[0].price;
    let mut prev_sign = 0i8;
    let mut imbalance = 0.0;

    for (i, trade) in trades.iter().enumerate().skip(1) {
        let sign = trade_sign(trade.price, prev_price, prev_sign);
        if sign != 0 {
            prev_sign = sign;
            let weight = match bar_type {
                ImbalanceBarType::Tick => 1.0,
                ImbalanceBarType::Volume => trade.volume,
                ImbalanceBarType::Dollar => trade.price * trade.volume,
            };
            imbalance += sign as f64 * weight;
        }

        prev_price = trade.price;

        if imbalance.abs() >= threshold {
            bars.push(build_bar(&trades[start_idx..=i]));
            start_idx = i + 1;
            imbalance = 0.0;
            prev_sign = 0;
            if start_idx < trades.len() {
                prev_price = trades[start_idx].price;
            }
        }
    }

    Ok(bars)
}

/// `trades` is never empty: every caller passes `start_idx..=i` with `start_idx <= i`.
fn build_bar(trades: &[Trade]) -> StandardBar {
    let (first, last) = (&trades[0], &trades[trades.len() - 1]);
    let open = first.price;
    let close = last.price;
    let start_timestamp = first.timestamp;
    let timestamp = last.timestamp;
    let (high, low) = trades.iter().fold((f64::NEG_INFINITY, f64::INFINITY), |(h, l), trade| {
        (h.max(trade.price), l.min(trade.price))
    });
    let (volume, dollar_value) = trades.iter().fold((0.0, 0.0), |(v, d), trade| {
        let next_v = v + trade.volume;
        let next_d = d + trade.price * trade.volume;
        (next_v, next_d)
    });

    StandardBar {
        start_timestamp,
        timestamp,
        open,
        high,
        low,
        close,
        volume,
        dollar_value,
        tick_count: trades.len(),
    }
}

fn trade_sign(price: f64, prev_price: f64, prev_sign: i8) -> i8 {
    if price > prev_price {
        1
    } else if price < prev_price {
        -1
    } else {
        prev_sign
    }
}
