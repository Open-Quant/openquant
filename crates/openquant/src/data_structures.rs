//! Data-structure bars (subset of mlfinlab: standard and time bars for now).
//! Additional bar types (run/imbalance) can be layered on top of these core
//! primitives as we add fixtures and reference expectations.

use chrono::{Duration, NaiveDateTime};

/// Single trade input.
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
    pub timestamp: NaiveDateTime,
    pub price: f64,
    pub volume: f64,
}

/// Supported standard bar accumulation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardBarType {
    Tick,
    Volume,
    Dollar,
}

/// Bar output with OHLCV-like fields.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardBar {
    pub start_timestamp: NaiveDateTime,
    pub timestamp: NaiveDateTime,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub dollar_value: f64,
    pub tick_count: usize,
}

/// Signed-imbalance accumulation modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImbalanceBarType {
    Tick,
    Volume,
    Dollar,
}

/// Construct standard bars (tick/volume/dollar) from a stream of trades using a static threshold.
///
/// This mirrors the mlfinlab behavior of emitting a bar whenever the chosen metric
/// crosses the threshold and starting accumulation fresh afterward. Any trailing
/// partial bar that does not satisfy the threshold is dropped.
pub fn standard_bars(
    trades: &[Trade],
    threshold: f64,
    bar_type: StandardBarType,
) -> Vec<StandardBar> {
    assert!(threshold.is_sign_positive(), "threshold must be positive");

    let mut bars = Vec::new();
    if trades.is_empty() {
        return bars;
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

    bars
}

/// Construct time bars using a fixed interval. The interval applies from the start
/// timestamp of the current bar; the trade that crosses the interval boundary is
/// included in the closing bar, and accumulation restarts afterward.
pub fn time_bars(trades: &[Trade], interval: Duration) -> Vec<StandardBar> {
    assert!(interval.num_microseconds().unwrap_or(0) > 0, "interval must be positive");

    let mut bars = Vec::new();
    if trades.is_empty() {
        return bars;
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

    bars
}

/// Construct run bars by counting consecutive price-direction runs. A bar closes when
/// `threshold` consecutive moves occur in the same direction. Trailing partial bars
/// that have not met the threshold are dropped.
pub fn run_bars(trades: &[Trade], threshold: usize) -> Vec<StandardBar> {
    assert!(threshold > 0, "threshold must be positive");
    if trades.len() < 2 {
        return Vec::new();
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

    bars
}

/// Construct imbalance bars by accumulating signed imbalance (tick, volume, or dollar)
/// until the absolute imbalance crosses `threshold`. Trailing partial bars that have
/// not met the threshold are dropped.
pub fn imbalance_bars(
    trades: &[Trade],
    threshold: f64,
    bar_type: ImbalanceBarType,
) -> Vec<StandardBar> {
    assert!(threshold.is_sign_positive(), "threshold must be positive");
    if trades.len() < 2 {
        return Vec::new();
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

    bars
}

fn build_bar(trades: &[Trade]) -> StandardBar {
    assert!(!trades.is_empty(), "cannot build a bar from an empty trade slice");

    let open = trades.first().expect("non-empty slice").price;
    let close = trades.last().expect("non-empty slice").price;
    let start_timestamp = trades.first().expect("non-empty slice").timestamp;
    let timestamp = trades.last().expect("non-empty slice").timestamp;
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
