use pyo3::prelude::*;

use crate::helpers::{
    format_naive_datetime, pair_timestamps_values, parse_naive_datetimes, to_py_err,
};

/// Python-facing reserve bet-size row: `(timestamp, active_long, active_short, c_t, bet_size)`.
type ReserveRow = (String, f64, f64, f64, f64);

/// Convert predicted-class probabilities into bet sizes in `[-1, 1]`.
///
/// AFML Snippet 10.1. With `p` the probability of the predicted class and `K = num_classes`,
/// `z = (p - 1/K) / sqrt(p (1 - p))` and the size is `2 Phi(z) - 1`. A probability below
/// `1/K` gives a negative size (a bet against the side); exactly 0 or 1 gives -1 or +1.
///
/// Parameters
/// ----------
/// prob : list[float]
///     Probability of the predicted class for each prediction.
/// num_classes : int
///     Number of classes `K` of the classifier.
/// pred : list[float] | None, default None
///     Side of each prediction (typically `+1`/`-1` from a primary model); each size is
///     multiplied by it. It is zipped with `prob`, so a shorter `pred` truncates the output.
///
/// Returns
/// -------
/// list[float]
///     One bet size per probability.
#[pyfunction(name = "get_signal")]
#[pyo3(signature = (prob, num_classes, pred=None))]
fn bet_sizing_get_signal(prob: Vec<f64>, num_classes: usize, pred: Option<Vec<f64>>) -> Vec<f64> {
    openquant::bet_sizing::get_signal(&prob, num_classes, pred.as_deref())
}

/// Round bet sizes to multiples of `step_size` and clamp them to `[-1, 1]`.
///
/// AFML Snippet 10.3. Discretising avoids overtrading on tiny size changes. A `step_size` of
/// zero or less returns the input unchanged (not clamped).
///
/// Parameters
/// ----------
/// signal0 : list[float]
///     Bet sizes.
/// step_size : float
///     Discretisation step, e.g. 0.1.
///
/// Returns
/// -------
/// list[float]
///     The discretised sizes, same length as `signal0`.
#[pyfunction(name = "discrete_signal")]
fn bet_sizing_discrete_signal(signal0: Vec<f64>, step_size: f64) -> Vec<f64> {
    openquant::bet_sizing::discrete_signal(&signal0, step_size)
}

/// Bet size for a price divergence, using the curve named by `func`.
///
/// AFML Snippet 10.4. `"sigmoid"` gives `x / sqrt(w + x^2)` in `(-1, 1)`; `"power"` gives
/// `sgn(x) |x|^w` and needs `x` scaled into `[-1, 1]`.
///
/// Parameters
/// ----------
/// w_param : float
///     Curve parameter `w` (sigmoid width or power exponent), e.g. from `get_w`.
/// price_div : float
///     Price divergence `x = forecast - market`.
/// func : str
///     `"sigmoid"` or `"power"`.
///
/// Returns
/// -------
/// float
///     The bet size.
///
/// Raises
/// ------
/// ValueError
///     If `func` is not `"sigmoid"` or `"power"`, or `func` is `"power"` and `price_div` is
///     outside `[-1, 1]`.
#[pyfunction(name = "bet_size")]
fn bet_sizing_bet_size(w_param: f64, price_div: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::bet_size(w_param, price_div, &func).map_err(to_py_err)
}

/// Sigmoid bet size `x / sqrt(w + x^2)` for a price divergence `x`.
///
/// AFML Snippet 10.4. The result is in `(-1, 1)`.
///
/// Parameters
/// ----------
/// w_param : float
///     Sigmoid width `w`, e.g. from `get_w_sigmoid`.
/// price_div : float
///     Price divergence `x = forecast - market`.
///
/// Returns
/// -------
/// float
///     The bet size.
#[pyfunction(name = "bet_size_sigmoid")]
fn bet_sizing_bet_size_sigmoid(w_param: f64, price_div: f64) -> f64 {
    openquant::bet_sizing::bet_size_sigmoid(w_param, price_div)
}

/// Power bet size `sgn(x) |x|^w` for a price divergence `x` scaled into `[-1, 1]`.
///
/// Parameters
/// ----------
/// w_param : float
///     Power exponent `w`, e.g. from `get_w_power`.
/// price_div : float
///     Price divergence `x`, scaled into `[-1, 1]`.
///
/// Returns
/// -------
/// float
///     The bet size (0 when `price_div` is 0).
///
/// Raises
/// ------
/// ValueError
///     If `price_div` is outside `[-1, 1]`.
#[pyfunction(name = "bet_size_power")]
fn bet_sizing_bet_size_power(w_param: f64, price_div: f64) -> PyResult<f64> {
    openquant::bet_sizing::bet_size_power(w_param, price_div).map_err(to_py_err)
}

/// Market price at which the bet size equals `m_bet_size`, for the curve named by `func`.
///
/// Inverse of `bet_size` (AFML Snippet 10.4): see `inv_price_sigmoid` and `inv_price_power`.
///
/// Parameters
/// ----------
/// forecast_price : float
///     Forecast price `f`.
/// w_param : float
///     Curve parameter `w` (sigmoid width or power exponent).
/// m_bet_size : float
///     Bet size `m`, in `[-1, 1]`.
/// func : str
///     `"sigmoid"` or `"power"`.
///
/// Returns
/// -------
/// float
///     The implied market price.
///
/// Raises
/// ------
/// ValueError
///     If `func` is not `"sigmoid"` or `"power"`.
#[pyfunction(name = "inv_price")]
fn bet_sizing_inv_price(
    forecast_price: f64,
    w_param: f64,
    m_bet_size: f64,
    func: String,
) -> PyResult<f64> {
    openquant::bet_sizing::inv_price(forecast_price, w_param, m_bet_size, &func).map_err(to_py_err)
}

/// Market price at which the sigmoid bet size equals `m_bet_size`.
///
/// AFML Snippet 10.4. Returns `f - m sqrt(w / (1 - m^2))`, which is not finite at `|m| = 1`.
///
/// Parameters
/// ----------
/// forecast_price : float
///     Forecast price `f`.
/// w_param : float
///     Sigmoid width `w`.
/// m_bet_size : float
///     Bet size `m`, in `(-1, 1)`.
///
/// Returns
/// -------
/// float
///     The implied market price.
#[pyfunction(name = "inv_price_sigmoid")]
fn bet_sizing_inv_price_sigmoid(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    openquant::bet_sizing::inv_price_sigmoid(forecast_price, w_param, m_bet_size)
}

/// Market price at which the power bet size equals `m_bet_size`.
///
/// Returns `f - sgn(m) |m|^(1/w)`, or `f` when `m` is 0.
///
/// Parameters
/// ----------
/// forecast_price : float
///     Forecast price `f`.
/// w_param : float
///     Power exponent `w`.
/// m_bet_size : float
///     Bet size `m`, in `[-1, 1]`.
///
/// Returns
/// -------
/// float
///     The implied market price.
#[pyfunction(name = "inv_price_power")]
fn bet_sizing_inv_price_power(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    openquant::bet_sizing::inv_price_power(forecast_price, w_param, m_bet_size)
}

/// Calibrate the curve parameter `w` so that divergence `price_div` gives size `m_bet_size`.
///
/// Dispatches to `get_w_sigmoid` or `get_w_power` (AFML Snippet 10.4).
///
/// Parameters
/// ----------
/// price_div : float
///     Price divergence `x = forecast - market` to calibrate on.
/// m_bet_size : float
///     Bet size `x` should map to.
/// func : str
///     `"sigmoid"` or `"power"`.
///
/// Returns
/// -------
/// float
///     The curve parameter `w`.
///
/// Raises
/// ------
/// ValueError
///     If `func` is not `"sigmoid"` or `"power"`, or `func` is `"power"` and `price_div` is
///     outside `[-1, 1]`.
#[pyfunction(name = "get_w")]
fn bet_sizing_get_w(price_div: f64, m_bet_size: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::get_w(price_div, m_bet_size, &func).map_err(to_py_err)
}

/// Sigmoid width `w = x^2 (1/m^2 - 1)` such that divergence `x` gives size `m`.
///
/// AFML Snippet 10.4.
///
/// Parameters
/// ----------
/// price_div : float
///     Price divergence `x` to calibrate on.
/// m_bet_size : float
///     Bet size `m` that `x` should map to.
///
/// Returns
/// -------
/// float
///     The sigmoid width `w`.
#[pyfunction(name = "get_w_sigmoid")]
fn bet_sizing_get_w_sigmoid(price_div: f64, m_bet_size: f64) -> f64 {
    openquant::bet_sizing::get_w_sigmoid(price_div, m_bet_size)
}

/// Power exponent `w = ln(m / sgn(x)) / ln|x|` such that divergence `x` gives size `m`.
///
/// A negative result is floored at 0.
///
/// Parameters
/// ----------
/// price_div : float
///     Price divergence `x` to calibrate on, in `[-1, 1]`.
/// m_bet_size : float
///     Bet size `m` that `x` should map to.
///
/// Returns
/// -------
/// float
///     The power exponent `w` (at least 0).
///
/// Raises
/// ------
/// ValueError
///     If `price_div` is outside `[-1, 1]`.
#[pyfunction(name = "get_w_power")]
fn bet_sizing_get_w_power(price_div: f64, m_bet_size: f64) -> PyResult<f64> {
    openquant::bet_sizing::get_w_power(price_div, m_bet_size).map_err(to_py_err)
}

/// Target position for a forecast and market price, using the curve named by `func`.
///
/// AFML Snippet 10.4. Returns `trunc(m(f - m_p) * max_pos)`: the bet size of the divergence
/// times the maximum position, truncated toward zero to whole units.
///
/// Parameters
/// ----------
/// w : float
///     Curve parameter (sigmoid width or power exponent), e.g. from `get_w`.
/// f : float
///     Forecast price.
/// m_p : float
///     Current market price.
/// max_pos : float
///     Maximum absolute position.
/// func : str
///     `"sigmoid"` or `"power"`.
///
/// Returns
/// -------
/// float
///     The target position, a whole number.
///
/// Raises
/// ------
/// ValueError
///     If `func` is not `"sigmoid"` or `"power"`, or `func` is `"power"` and `f - m_p` is
///     outside `[-1, 1]`.
#[pyfunction(name = "get_target_pos")]
fn bet_sizing_get_target_pos(
    w: f64,
    f: f64,
    m_p: f64,
    max_pos: f64,
    func: String,
) -> PyResult<f64> {
    openquant::bet_sizing::get_target_pos(w, f, m_p, max_pos, &func).map_err(to_py_err)
}

/// Sigmoid target position `trunc(m(forecast - market) * max_pos)`.
///
/// AFML Snippet 10.4. The result is truncated toward zero to whole units.
///
/// Parameters
/// ----------
/// w_param : float
///     Sigmoid width `w`.
/// forecast_price : float
///     Forecast price.
/// market_price : float
///     Current market price.
/// max_pos : float
///     Maximum absolute position.
///
/// Returns
/// -------
/// float
///     The target position, a whole number.
#[pyfunction(name = "get_target_pos_sigmoid")]
fn bet_sizing_get_target_pos_sigmoid(
    w_param: f64,
    forecast_price: f64,
    market_price: f64,
    max_pos: f64,
) -> f64 {
    openquant::bet_sizing::get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos)
}

/// Power-curve target position `trunc(m(forecast - market) * max_pos)`.
///
/// The result is truncated toward zero to whole units.
///
/// Parameters
/// ----------
/// w_param : float
///     Power exponent `w`.
/// forecast_price : float
///     Forecast price.
/// market_price : float
///     Current market price; `forecast_price - market_price` must lie in `[-1, 1]`.
/// max_pos : float
///     Maximum absolute position.
///
/// Returns
/// -------
/// float
///     The target position, a whole number.
///
/// Raises
/// ------
/// ValueError
///     If `forecast_price - market_price` is outside `[-1, 1]`.
#[pyfunction(name = "get_target_pos_power")]
fn bet_sizing_get_target_pos_power(
    w_param: f64,
    forecast_price: f64,
    market_price: f64,
    max_pos: f64,
) -> PyResult<f64> {
    openquant::bet_sizing::get_target_pos_power(w_param, forecast_price, market_price, max_pos)
        .map_err(to_py_err)
}

/// Breakeven limit price for moving from `pos` to `t_pos`, using the curve named by `func`.
///
/// AFML Snippet 10.4, extended to every direction. Both positions are truncated to whole
/// units; the result is the mean of `inv_price(f, w, k / max_pos)` over the positions `k`
/// passed through on the way (`pos + 1, ..., t_pos` when increasing), signs kept. Unlike the
/// snippet, this is also correct when reducing, going short or crossing zero. Returns NaN when
/// the truncated target equals the truncated current position.
///
/// Parameters
/// ----------
/// t_pos : float
///     Target position.
/// pos : float
///     Current position.
/// f : float
///     Forecast price.
/// w : float
///     Curve parameter (sigmoid width or power exponent).
/// max_pos : float
///     Maximum absolute position.
/// func : str
///     `"sigmoid"` or `"power"`.
///
/// Returns
/// -------
/// float
///     The limit price, or NaN when there is nothing to trade.
///
/// Raises
/// ------
/// ValueError
///     If `func` is not `"sigmoid"` or `"power"`.
#[pyfunction(name = "limit_price")]
fn bet_sizing_limit_price(
    t_pos: f64,
    pos: f64,
    f: f64,
    w: f64,
    max_pos: f64,
    func: String,
) -> PyResult<f64> {
    openquant::bet_sizing::limit_price(t_pos, pos, f, w, max_pos, &func).map_err(to_py_err)
}

/// Sigmoid breakeven limit price for moving from `pos` to `t_pos`.
///
/// AFML Snippet 10.4, extended to every direction. Both positions are truncated to whole
/// units; the result is the mean of `inv_price_sigmoid(f, w, k / max_pos)` over the positions
/// `k` passed through on the way, signs kept, so the limit price of a short lies above `f`.
/// Unlike the snippet, this is also correct when reducing, going short or crossing zero.
/// Returns NaN when the truncated target equals the truncated current position.
///
/// Parameters
/// ----------
/// t_pos : float
///     Target position.
/// pos : float
///     Current position.
/// f : float
///     Forecast price.
/// w : float
///     Sigmoid width.
/// max_pos : float
///     Maximum absolute position.
///
/// Returns
/// -------
/// float
///     The limit price, or NaN when there is nothing to trade.
#[pyfunction(name = "limit_price_sigmoid")]
fn bet_sizing_limit_price_sigmoid(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    openquant::bet_sizing::limit_price_sigmoid(t_pos, pos, f, w, max_pos)
}

/// Power-curve breakeven limit price for moving from `pos` to `t_pos`.
///
/// Same convention as `limit_price_sigmoid`: the mean of `inv_price_power(f, w, k / max_pos)`
/// over the whole-unit positions `k` passed through on the way, signs kept. Returns NaN when
/// the truncated target equals the truncated current position.
///
/// Parameters
/// ----------
/// t_pos : float
///     Target position.
/// pos : float
///     Current position.
/// f : float
///     Forecast price.
/// w : float
///     Power exponent.
/// max_pos : float
///     Maximum absolute position.
///
/// Returns
/// -------
/// float
///     The limit price, or NaN when there is nothing to trade.
#[pyfunction(name = "limit_price_power")]
fn bet_sizing_limit_price_power(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    openquant::bet_sizing::limit_price_power(t_pos, pos, f, w, max_pos)
}

/// Average the signals that are live at each change point.
///
/// AFML Snippet 10.2. A signal is live on `[start, end)`. The evaluation points are the
/// sorted, unique union of all starts and ends; at each, the result is the mean size of the
/// live signals, or 0 if none is live.
///
/// Parameters
/// ----------
/// signal_timestamps : list[str]
///     Signal start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// signal_values : list[float]
///     Bet size of each signal.
/// t1_timestamps : list[str]
///     End timestamp of each signal. It is zipped with the signals, so extra entries in the
///     longer list are ignored.
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     `(timestamp, average_size)` rows in increasing time.
///
/// Raises
/// ------
/// ValueError
///     If `signal_timestamps` and `signal_values` differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "avg_active_signals")]
fn bet_sizing_avg_active_signals(
    signal_timestamps: Vec<String>,
    signal_values: Vec<f64>,
    t1_timestamps: Vec<String>,
) -> PyResult<Vec<(String, f64)>> {
    let signal = pair_timestamps_values(
        signal_timestamps,
        signal_values,
        "signal_timestamps",
        "signal_values",
    )?;
    let t1 = parse_naive_datetimes(t1_timestamps)?;
    let result = openquant::bet_sizing::avg_active_signals(&signal, &t1);
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

/// Dynamic bet sizing from a price forecast.
///
/// AFML 10.6, Snippet 10.4, with the sigmoid curve. Inputs are broadcast to a common length
/// (the longest); each must have that length or length 1. The width is fixed at
/// `get_w_sigmoid(10.0, 0.95)`: a divergence of 10 price units gives a size of 0.95, whatever
/// the instrument. For anything else calibrate `w` with `get_w` and use `get_target_pos` and
/// `limit_price`.
///
/// Parameters
/// ----------
/// pos : list[float]
///     Current position.
/// max_pos : list[float]
///     Maximum absolute position.
/// m_p : list[float]
///     Current market price.
/// f : list[float]
///     Forecast price.
///
/// Returns
/// -------
/// list[tuple[float, float, float]]
///     One `(bet_size, target_position, limit_price)` row per broadcast row; `limit_price`
///     is NaN when the target equals the current position.
///
/// Raises
/// ------
/// ValueError
///     If every input is empty, or an input has neither length 1 nor the common length.
#[pyfunction(name = "bet_size_dynamic")]
fn bet_sizing_bet_size_dynamic(
    pos: Vec<f64>,
    max_pos: Vec<f64>,
    m_p: Vec<f64>,
    f: Vec<f64>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    openquant::bet_sizing::bet_size_dynamic(&pos, &max_pos, &m_p, &f).map_err(to_py_err)
}

/// CDF at `x` of the two-Gaussian mixture `p1 N(mu1, sigma1) + (1 - p1) N(mu2, sigma2)`.
///
/// Standard deviations are floored at `1e-8`. `x` may be infinite.
///
/// Parameters
/// ----------
/// mu1 : float
///     Mean of the first component.
/// mu2 : float
///     Mean of the second component.
/// sigma1 : float
///     Standard deviation of the first component.
/// sigma2 : float
///     Standard deviation of the second component.
/// p1 : float
///     Weight of the first component.
/// x : float
///     Point at which to evaluate the CDF.
///
/// Returns
/// -------
/// float
///     The mixture CDF at `x`.
///
/// Raises
/// ------
/// ValueError
///     If a mixture parameter is not finite or `x` is NaN.
#[pyfunction(name = "cdf_mixture")]
fn bet_sizing_cdf_mixture(
    mu1: f64,
    mu2: f64,
    sigma1: f64,
    sigma2: f64,
    p1: f64,
    x: f64,
) -> PyResult<f64> {
    openquant::bet_sizing::cdf_mixture(mu1, mu2, sigma1, sigma2, p1, x).map_err(to_py_err)
}

/// Reserve bet size for net concurrency `c` under a fitted two-Gaussian mixture.
///
/// AFML 10.2. With `F` the mixture CDF (`cdf_mixture`), returns `(F(c) - F(0)) / (1 - F(0))`
/// for `c >= 0` and `(F(c) - F(0)) / F(0)` otherwise.
///
/// Parameters
/// ----------
/// c : float
///     Net concurrency (active longs minus active shorts).
/// fit : list[float]
///     Mixture parameters `[mu1, mu2, sigma1, sigma2, p1]` (exactly five values).
///
/// Returns
/// -------
/// float
///     The bet size, in `[-1, 1]`.
///
/// Raises
/// ------
/// ValueError
///     If `fit` does not have exactly five values, a value of `fit` is not finite, or `c` is
///     NaN.
#[pyfunction(name = "single_bet_size_mixed")]
fn bet_sizing_single_bet_size_mixed(c: f64, fit: [f64; 5]) -> PyResult<f64> {
    openquant::bet_sizing::single_bet_size_mixed(c, &fit).map_err(to_py_err)
}

/// Count the long and short bets live at each bet's start.
///
/// AFML 10.2. A bet is live on `[start, end)`. For each bet's start `t`, counts the bets with
/// `start <= t < end`, split by side: `side > 0` is long and anything else (including 0) is
/// short.
///
/// Parameters
/// ----------
/// t1_starts : list[str]
///     Bet start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// t1_ends : list[str]
///     Bet end timestamps, in the same format.
/// side : list[float]
///     Direction of each bet.
///
/// Returns
/// -------
/// list[tuple[str, float, float]]
///     `(start, active_long, active_short)` rows, in input order.
///
/// Raises
/// ------
/// ValueError
///     If `t1_starts`, `t1_ends` and `side` differ in length, or a timestamp does not parse.
#[pyfunction(name = "get_concurrent_sides")]
fn bet_sizing_get_concurrent_sides(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
) -> PyResult<Vec<(String, f64, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t1_starts/t1_ends/side length mismatch",
        ));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::get_concurrent_sides(&t1, &side).map_err(to_py_err)?;
    Ok(result
        .into_iter()
        .map(|(ts, long, short)| (format_naive_datetime(&ts), long, short))
        .collect())
}

/// Budgeting bet size `L_t / max(L) - S_t / max(S)` from concurrent bet counts.
///
/// AFML 10.2. `L_t` and `S_t` are the long and short bets live at each bet's start (see
/// `get_concurrent_sides`); a side with no bets contributes 0. The maxima are taken over the
/// whole input, so computing this over a full backtest and trading on it looks ahead.
///
/// Parameters
/// ----------
/// t1_starts : list[str]
///     Bet start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// t1_ends : list[str]
///     Bet end timestamps, in the same format.
/// side : list[float]
///     Direction of each bet (`> 0` long, otherwise short).
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     `(start, bet_size)` rows, in input order.
///
/// Raises
/// ------
/// ValueError
///     If `t1_starts`, `t1_ends` and `side` differ in length, or a timestamp does not parse.
#[pyfunction(name = "bet_size_budget")]
fn bet_sizing_bet_size_budget(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
) -> PyResult<Vec<(String, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t1_starts/t1_ends/side length mismatch",
        ));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::bet_size_budget(&t1, &side).map_err(to_py_err)?;
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

/// Size bets from classifier probabilities.
///
/// AFML 10.3-10.5, Snippets 10.1-10.3: `get_signal` (signed by `sides`), then optionally
/// `avg_active_signals`, then `discrete_signal` with step `abs(step_size)` (0 disables
/// discretisation).
///
/// Parameters
/// ----------
/// event_starts : list[str]
///     Event start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// event_ends : list[str]
///     Event end timestamps (`t1`), in the same format.
/// probs : list[float]
///     Probability of the predicted class for each event.
/// sides : list[float]
///     Side of each event (typically `+1`/`-1`).
/// num_classes : int
///     Number of classes of the classifier.
/// step_size : float
///     Discretisation step; its absolute value is used.
/// average_active : bool
///     Average the sizes of concurrently live bets at each change point.
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     `(timestamp, bet_size)` rows: one per event (at its start) without averaging, or one
///     per change point in increasing time with `average_active=True`.
///
/// Raises
/// ------
/// ValueError
///     If `event_starts`, `event_ends`, `probs` and `sides` differ in length, or a timestamp
///     does not parse.
#[pyfunction(name = "bet_size_probability")]
fn bet_sizing_bet_size_probability(
    event_starts: Vec<String>,
    event_ends: Vec<String>,
    probs: Vec<f64>,
    sides: Vec<f64>,
    num_classes: usize,
    step_size: f64,
    average_active: bool,
) -> PyResult<Vec<(String, f64)>> {
    let starts = parse_naive_datetimes(event_starts)?;
    let ends = parse_naive_datetimes(event_ends)?;
    if starts.len() != ends.len() || starts.len() != probs.len() || starts.len() != sides.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "event_starts/event_ends/probs/sides length mismatch",
        ));
    }
    let events: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime, f64, f64)> = starts
        .into_iter()
        .zip(ends)
        .zip(probs)
        .zip(sides)
        .map(|(((s, e), p), sd)| (s, e, p, sd))
        .collect();
    let result = openquant::bet_sizing::bet_size_probability(
        &events,
        num_classes,
        step_size,
        average_active,
    );
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

/// Average the live signals at each given timestamp.
///
/// The worker behind `avg_active_signals` (AFML Snippet 10.2). At each `t` in
/// `molecule_timestamps`, returns the mean size of the signals with `start <= t < end`, or 0
/// if none is live.
///
/// Parameters
/// ----------
/// signal_timestamps : list[str]
///     Signal start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// signal_values : list[float]
///     Bet size of each signal.
/// t1_timestamps : list[str]
///     End timestamp of each signal (zipped with the signals; extra entries are ignored).
/// molecule_timestamps : list[str]
///     Timestamps at which to evaluate the average.
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     `(timestamp, average_size)` rows, in `molecule_timestamps` order.
///
/// Raises
/// ------
/// ValueError
///     If `signal_timestamps` and `signal_values` differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "mp_avg_active_signals")]
fn bet_sizing_mp_avg_active_signals(
    signal_timestamps: Vec<String>,
    signal_values: Vec<f64>,
    t1_timestamps: Vec<String>,
    molecule_timestamps: Vec<String>,
) -> PyResult<Vec<(String, f64)>> {
    let signal = pair_timestamps_values(
        signal_timestamps,
        signal_values,
        "signal_timestamps",
        "signal_values",
    )?;
    let t1 = parse_naive_datetimes(t1_timestamps)?;
    let molecule = parse_naive_datetimes(molecule_timestamps)?;
    let result = openquant::bet_sizing::mp_avg_active_signals(&signal, &t1, &molecule);
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

/// Reserve bet sizes under a given two-Gaussian mixture fit.
///
/// AFML 10.2. Computes `c_t = active_long - active_short` at each bet's start (see
/// `get_concurrent_sides`) and sizes it with `single_bet_size_mixed`. Pass the parameters
/// returned by `bet_size_reserve_full` for reproducible sizes.
///
/// Parameters
/// ----------
/// t1_starts : list[str]
///     Bet start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// t1_ends : list[str]
///     Bet end timestamps, in the same format.
/// side : list[float]
///     Direction of each bet (`> 0` long, otherwise short).
/// fit : list[float]
///     Mixture parameters `[mu1, mu2, sigma1, sigma2, p1]` (exactly five values).
///
/// Returns
/// -------
/// list[tuple[str, float, float, float]]
///     `(start, active_long, active_short, bet_size)` rows, in input order.
///
/// Raises
/// ------
/// ValueError
///     If `t1_starts`, `t1_ends` and `side` differ in length, a timestamp does not parse, or
///     `fit` does not have exactly five finite values.
#[pyfunction(name = "bet_size_reserve")]
fn bet_sizing_bet_size_reserve(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
    fit: [f64; 5],
) -> PyResult<Vec<(String, f64, f64, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t1_starts/t1_ends/side length mismatch",
        ));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::bet_size_reserve(&t1, &side, &fit).map_err(to_py_err)?;
    Ok(result.into_iter().map(|(ts, l, s, b)| (format_naive_datetime(&ts), l, s, b)).collect())
}

/// Reserve bet sizes under a given mixture fit, keeping the net concurrency in each row.
///
/// Like `bet_size_reserve`, with the `c_t = active_long - active_short` column included.
///
/// Parameters
/// ----------
/// t1_starts : list[str]
///     Bet start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// t1_ends : list[str]
///     Bet end timestamps, in the same format.
/// side : list[float]
///     Direction of each bet (`> 0` long, otherwise short).
/// fit : list[float]
///     Mixture parameters `[mu1, mu2, sigma1, sigma2, p1]` (exactly five values).
///
/// Returns
/// -------
/// list[tuple[str, float, float, float, float]]
///     `(start, active_long, active_short, c_t, bet_size)` rows, in input order.
///
/// Raises
/// ------
/// ValueError
///     If `t1_starts`, `t1_ends` and `side` differ in length, a timestamp does not parse, or
///     `fit` does not have exactly five finite values.
#[pyfunction(name = "bet_size_reserve_with_fit")]
fn bet_sizing_bet_size_reserve_with_fit(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
    fit: [f64; 5],
) -> PyResult<Vec<ReserveRow>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t1_starts/t1_ends/side length mismatch",
        ));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result =
        openquant::bet_sizing::bet_size_reserve_with_fit(&t1, &side, &fit).map_err(to_py_err)?;
    Ok(result
        .into_iter()
        .map(|(ts, l, s, c, b)| (format_naive_datetime(&ts), l, s, c, b))
        .collect())
}

/// Reserve bet sizing with a two-Gaussian mixture fitted to the net concurrency.
///
/// AFML 10.2. Computes `c_t = active_long - active_short` at each bet's start, fits a
/// two-Gaussian mixture to it by EM (`fit_runs` random restarts, each stopping when the
/// log-likelihood changes by less than `epsilon` or after `max_iter` iterations), and sizes
/// each bet with `single_bet_size_mixed`. AFML fits the mixture with EF3M moment matching
/// instead. The EM starts are random (unseeded), so results change from run to run; pass
/// `return_parameters=True` and reuse the fit with `bet_size_reserve` for reproducible sizes.
///
/// Parameters
/// ----------
/// t1_starts : list[str]
///     Bet start timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// t1_ends : list[str]
///     Bet end timestamps, in the same format.
/// side : list[float]
///     Direction of each bet (`> 0` long, otherwise short).
/// fit_runs : int
///     Number of EM restarts (at least one is run).
/// epsilon : float
///     Log-likelihood change below which an EM run stops.
/// max_iter : int
///     Maximum EM iterations per run (at least one is run).
/// return_parameters : bool
///     Also return the fitted mixture parameters.
///
/// Returns
/// -------
/// tuple[list[tuple[str, float, float, float, float]], list[float] | None]
///     `(rows, params)`: `(start, active_long, active_short, c_t, bet_size)` rows in input
///     order, and `[mu1, mu2, sigma1, sigma2, p1]` when `return_parameters` is True, else None.
///
/// Raises
/// ------
/// ValueError
///     If the inputs are empty, `t1_starts`, `t1_ends` and `side` differ in length, or a
///     timestamp does not parse.
#[pyfunction(name = "bet_size_reserve_full")]
fn bet_sizing_bet_size_reserve_full(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
    fit_runs: usize,
    epsilon: f64,
    max_iter: usize,
    return_parameters: bool,
) -> PyResult<(Vec<ReserveRow>, Option<[f64; 5]>)> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t1_starts/t1_ends/side length mismatch",
        ));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let (events, params) = openquant::bet_sizing::bet_size_reserve_full(
        &t1,
        &side,
        fit_runs,
        epsilon,
        max_iter,
        return_parameters,
    )
    .map_err(to_py_err)?;
    let out_events = events
        .into_iter()
        .map(|(ts, l, s, c, b)| (format_naive_datetime(&ts), l, s, c, b))
        .collect();
    Ok((out_events, params))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "bet_sizing")?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_signal, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_discrete_signal, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_inv_price, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_inv_price_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_inv_price_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_w, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_w_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_w_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_target_pos, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_target_pos_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_target_pos_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_limit_price, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_limit_price_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_limit_price_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_avg_active_signals, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_dynamic, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_cdf_mixture, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_single_bet_size_mixed, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_concurrent_sides, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_budget, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_probability, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_mp_avg_active_signals, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_reserve, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_reserve_with_fit, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_reserve_full, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("bet_sizing", m)?;
    Ok(())
}
