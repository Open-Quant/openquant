//! The ETF trick and futures roll gaps (AFML §2.4.1 and §2.4.3, Snippet 2.2).
//!
//! [`EtfTrick`] turns a basket of instruments with changing allocations, rolls, FX rates and
//! carry into the value `K_t` of one dollar invested in it: a synthetic total-return series
//! with no gaps at rolls or rebalances, which can be fed to bars, filters and labels like a
//! single instrument. [`get_futures_roll_series`] is the single-contract special case: the
//! cumulative roll gap of one futures chain.
//!
//! Conventions:
//!
//! - [`Table`]s share one row index (oldest first) and one set of columns (instruments).
//!   Columns are matched by name to the allocation table's order.
//! - Allocations are de-levered by the sum of their absolute values, and holdings are sized
//!   at the **next** bar's open; the bar after a rebalance earns open-to-close only.
//! - `costs` holds carry or dividends in price units with the sign of a credit: it is
//!   **added** to the price change. Transaction costs are not modelled, so `K_t` is gross.
//! - A rebalance is detected by exact equality of consecutive allocation rows.
//!
//! This module is Rust-only; there is no Python binding.
//!
//! ```
//! use openquant::etf_trick::{EtfTrick, Table};
//!
//! # fn main() -> Result<(), openquant::etf_trick::EtfTrickError> {
//! let table = |values: [[f64; 2]; 6]| Table {
//!     index: ["01-02", "01-03", "01-04", "01-05", "01-08", "01-09"].map(String::from).to_vec(),
//!     columns: vec!["CL".to_string(), "NG".to_string()],
//!     values: values.iter().map(|row| row.to_vec()).collect(),
//! };
//! let open = [[70.0, 2.50], [70.5, 2.52], [71.4, 2.49], [71.0, 2.55], [72.2, 2.60], [72.0, 2.58]];
//! let close =
//!     [[70.4, 2.51], [71.2, 2.50], [71.1, 2.54], [72.0, 2.61], [72.1, 2.57], [72.6, 2.59]];
//! let alloc = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]];
//!
//! let etf = EtfTrick::from_tables(
//!     table(open),
//!     table(close),
//!     table(alloc),
//!     table([[0.0; 2]; 6]), // no carry
//!     None,                 // no FX
//! )?;
//! let series = etf.get_etf_series(100)?;
//! // Six rows give four values: the first seeds the previous close, the last lacks a next open.
//! assert_eq!(series.len(), 4);
//! assert_eq!(series[0], ("01-03".to_string(), 1.0));
//! // 1 + 0.5 / 71.4 * (-0.10) + 0.5 / 2.49 * 0.04
//! assert!((series[1].1 - 1.007332).abs() < 1e-6);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use std::collections::HashMap;
use std::path::Path;

use chrono::NaiveDate;
use csv::StringRecord;

/// Errors returned by the ETF trick and roll-gap functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EtfTrickError {
    /// Reading a CSV failed; `reason` is the underlying I/O or parse error.
    #[error("failed to {action} {path}: {reason}")]
    Csv {
        /// What was being done (`"open"`, `"read headers"`, `"read record"`).
        action: &'static str,
        /// The file.
        path: String,
        /// The underlying error message.
        reason: String,
    },
    /// A CSV has fewer than two columns (an index and at least one value column).
    #[error("csv {path} must have at least index + 1 value column")]
    TooFewColumns {
        /// The file.
        path: String,
    },
    /// A CSV record has no index cell.
    #[error("missing index column in {path}")]
    MissingIndexColumn {
        /// The file.
        path: String,
    },
    /// A CSV value cell is not a number.
    #[error("failed to parse float '{cell}' in {path}: {reason}")]
    ParseFloat {
        /// The cell text.
        cell: String,
        /// The file.
        path: String,
        /// The parse error message.
        reason: String,
    },
    /// A table lacks a column named in the allocation table.
    #[error("missing column '{0}' in table")]
    MissingColumn(String),
    /// `batch_size` is below 3 for a CSV source.
    #[error("Batch size should be >= 3")]
    BatchTooSmall,
    /// The tables do not share the same row index.
    #[error("DataFrames indices are different")]
    IndexMismatch,
    /// The tables have different numbers of columns.
    #[error("DataFrames columns are different")]
    ColumnMismatch,
    /// Internal: holdings were not initialised before use.
    #[error("missing previous h")]
    MissingPreviousHoldings,
    /// The roll method is not `"absolute"` or `"relative"`.
    #[error("The method must be either absolute or relative, Check spelling.")]
    UnknownRollMethod,
}

impl EtfTrickError {
    fn csv(action: &'static str, path: &Path, err: csv::Error) -> Self {
        Self::Csv { action, path: path.display().to_string(), reason: err.to_string() }
    }
}

/// A small labelled matrix: one row per date, one column per instrument.
#[derive(Clone, Debug)]
pub struct Table {
    /// Row labels (typically dates), oldest first.
    pub index: Vec<String>,
    /// Column names (instruments).
    pub columns: Vec<String>,
    /// Row-major values; each row has one value per column.
    pub values: Vec<Vec<f64>>,
}

impl Table {
    /// Reads a table from a CSV with a header row: the first column is the index and every
    /// other column a numeric value column.
    ///
    /// # Errors
    ///
    /// - [`EtfTrickError::Csv`] if the file cannot be opened or a record cannot be read.
    /// - [`EtfTrickError::TooFewColumns`] if the header has fewer than two columns.
    /// - [`EtfTrickError::MissingIndexColumn`] if a record is empty.
    /// - [`EtfTrickError::ParseFloat`] if a value cell is not a number.
    pub fn from_csv(path: &Path) -> Result<Self, EtfTrickError> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| EtfTrickError::csv("open", path, e))?;
        let headers =
            rdr.headers().map_err(|e| EtfTrickError::csv("read headers", path, e))?.clone();

        if headers.len() < 2 {
            return Err(EtfTrickError::TooFewColumns { path: path.display().to_string() });
        }

        let columns = headers.iter().skip(1).map(ToString::to_string).collect::<Vec<_>>();
        let mut index = Vec::new();
        let mut values = Vec::new();

        for rec in rdr.records() {
            let record = rec.map_err(|e| EtfTrickError::csv("read record", path, e))?;
            let (idx, row) = parse_row(&record, path)?;
            index.push(idx);
            values.push(row);
        }

        Ok(Self { index, columns, values })
    }

    fn align_columns(&self, ordered_columns: &[String]) -> Result<Self, EtfTrickError> {
        let mut col_to_idx = HashMap::new();
        for (i, c) in self.columns.iter().enumerate() {
            col_to_idx.insert(c.as_str(), i);
        }

        let mut aligned_values = Vec::with_capacity(self.values.len());
        for row in &self.values {
            let mut out_row = Vec::with_capacity(ordered_columns.len());
            for c in ordered_columns {
                let idx = col_to_idx
                    .get(c.as_str())
                    .ok_or_else(|| EtfTrickError::MissingColumn(c.clone()))?;
                out_row.push(row[*idx]);
            }
            aligned_values.push(out_row);
        }

        Ok(Self {
            index: self.index.clone(),
            columns: ordered_columns.to_vec(),
            values: aligned_values,
        })
    }
}

/// The ETF trick (AFML §2.4.1): the value of one dollar invested in a basket.
///
/// Build from in-memory [`Table`]s with [`EtfTrick::from_tables`] or from CSV paths with
/// [`EtfTrick::from_csv`], then call [`EtfTrick::get_etf_series`].
#[derive(Clone, Debug)]
pub struct EtfTrick {
    source: Source,
}

#[derive(Clone, Debug)]
struct InMemoryTables {
    open: Table,
    close: Table,
    alloc: Table,
    costs: Table,
    rates: Option<Table>,
}

#[derive(Clone, Debug)]
enum Source {
    // Boxed so the enum is not sized by the five in-memory tables.
    InMemory(Box<InMemoryTables>),
    Csv {
        open_path: String,
        close_path: String,
        alloc_path: String,
        costs_path: String,
        rates_path: Option<String>,
    },
}

impl EtfTrick {
    /// Creates an ETF trick from in-memory tables.
    ///
    /// `open` and `close` are prices, `alloc` the target weights (any leverage; they are
    /// de-levered), `costs` the carry or dividends per bar in price units (added to the price
    /// change), and `rates` the FX rate of each instrument to the account currency (1 when
    /// `None`). All tables must share the row index and column count.
    ///
    /// # Errors
    ///
    /// - [`EtfTrickError::IndexMismatch`] if a table's index or row count differs from
    ///   `open`'s.
    /// - [`EtfTrickError::ColumnMismatch`] if a table's column count differs from `open`'s.
    pub fn from_tables(
        open: Table,
        close: Table,
        alloc: Table,
        costs: Table,
        rates: Option<Table>,
    ) -> Result<Self, EtfTrickError> {
        validate_shapes(&open, &close, &alloc, &costs, rates.as_ref())?;
        Ok(Self {
            source: Source::InMemory(Box::new(InMemoryTables { open, close, alloc, costs, rates })),
        })
    }

    /// Creates an ETF trick that reads its tables from CSV files (see [`Table::from_csv`]) when
    /// [`EtfTrick::get_etf_series`] is called.
    ///
    /// # Errors
    ///
    /// Never; the files are not opened until [`EtfTrick::get_etf_series`]. The `Result` is
    /// kept for API stability.
    pub fn from_csv(
        open_path: &str,
        close_path: &str,
        alloc_path: &str,
        costs_path: &str,
        rates_path: Option<&str>,
    ) -> Result<Self, EtfTrickError> {
        Ok(Self {
            source: Source::Csv {
                open_path: open_path.to_string(),
                close_path: close_path.to_string(),
                alloc_path: alloc_path.to_string(),
                costs_path: costs_path.to_string(),
                rates_path: rates_path.map(ToString::to_string),
            },
        })
    }

    /// Computes the value series `(index, K_t)`, starting from `K = 1`.
    ///
    /// Holdings at a rebalance are `h = w K / (o_{t+1} fx sum|w|)`; each later bar adds
    /// `sum h fx (delta + costs)`, with `delta` the close-to-close change, or open-to-close on
    /// the bar after a rebalance. The first row only seeds the previous close and the last
    /// row is dropped (sizing there needs a next open), so `n` rows give `n - 2` values;
    /// fewer than two rows give none. This matches mlfinlab's output.
    ///
    /// `batch_size` exists for mlfinlab compatibility: it is checked for CSV sources and
    /// otherwise ignored. CSV files are read in full.
    ///
    /// # Errors
    ///
    /// - [`EtfTrickError::BatchTooSmall`] if the source is CSV and `batch_size < 3`.
    /// - Any [`Table::from_csv`] error for a CSV source.
    /// - [`EtfTrickError::IndexMismatch`] or [`EtfTrickError::ColumnMismatch`] if the tables
    ///   disagree in shape.
    /// - [`EtfTrickError::MissingColumn`] if a table lacks an instrument named in `alloc`.
    pub fn get_etf_series(&self, batch_size: usize) -> Result<Vec<(String, f64)>, EtfTrickError> {
        match &self.source {
            Source::InMemory(tables) => compute_etf_series(
                &tables.open,
                &tables.close,
                &tables.alloc,
                &tables.costs,
                tables.rates.as_ref(),
            ),
            Source::Csv { open_path, close_path, alloc_path, costs_path, rates_path } => {
                if batch_size < 3 {
                    return Err(EtfTrickError::BatchTooSmall);
                }

                let open = Table::from_csv(Path::new(open_path))?;
                let close = Table::from_csv(Path::new(close_path))?;
                let alloc = Table::from_csv(Path::new(alloc_path))?;
                let costs = Table::from_csv(Path::new(costs_path))?;
                let rates = if let Some(rp) = rates_path {
                    Some(Table::from_csv(Path::new(rp))?)
                } else {
                    None
                };

                compute_etf_series(&open, &close, &alloc, &costs, rates.as_ref())
            }
        }
    }

    /// Does nothing; kept for mlfinlab API compatibility (the computation holds no state
    /// between calls).
    pub fn reset(&mut self) {}
}

fn parse_row(record: &StringRecord, path: &Path) -> Result<(String, Vec<f64>), EtfTrickError> {
    let idx = record
        .get(0)
        .ok_or_else(|| EtfTrickError::MissingIndexColumn { path: path.display().to_string() })?
        .to_string();
    let mut row = Vec::with_capacity(record.len().saturating_sub(1));
    for cell in record.iter().skip(1) {
        let v = cell.parse::<f64>().map_err(|e| EtfTrickError::ParseFloat {
            cell: cell.to_string(),
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        row.push(v);
    }
    Ok((idx, row))
}

fn validate_shapes(
    open: &Table,
    close: &Table,
    alloc: &Table,
    costs: &Table,
    rates: Option<&Table>,
) -> Result<(), EtfTrickError> {
    let pairs = [close, alloc, costs];
    for t in pairs {
        if open.index != t.index || open.values.len() != t.values.len() {
            return Err(EtfTrickError::IndexMismatch);
        }
        if open.columns.len() != t.columns.len() {
            return Err(EtfTrickError::ColumnMismatch);
        }
    }
    if let Some(r) = rates {
        if open.index != r.index || open.values.len() != r.values.len() {
            return Err(EtfTrickError::IndexMismatch);
        }
        if open.columns.len() != r.columns.len() {
            return Err(EtfTrickError::ColumnMismatch);
        }
    }
    Ok(())
}

fn compute_etf_series(
    open: &Table,
    close: &Table,
    alloc: &Table,
    costs: &Table,
    rates: Option<&Table>,
) -> Result<Vec<(String, f64)>, EtfTrickError> {
    validate_shapes(open, close, alloc, costs, rates)?;

    let securities = alloc.columns.clone();
    let open = open.align_columns(&securities)?;
    let close = close.align_columns(&securities)?;
    let alloc = alloc.align_columns(&securities)?;
    let costs = costs.align_columns(&securities)?;
    let rates = if let Some(r) = rates {
        r.align_columns(&securities)?
    } else {
        Table {
            index: open.index.clone(),
            columns: securities.clone(),
            values: vec![vec![1.0; securities.len()]; open.values.len()],
        }
    };

    if open.values.len() < 2 {
        return Ok(Vec::new());
    }

    let n_rows = open.values.len();
    let n_cols = securities.len();
    let mut out = Vec::new();

    let mut prev_k = 1.0f64;
    let mut prev_allocs_change = false;
    let mut prev_h: Option<Vec<f64>> = None;
    let mut prev_allocs = vec![f64::NAN; n_cols];

    // Last row needs next-open for h_t and is effectively omitted in mlfinlab output.
    for i in 1..(n_rows - 1) {
        let weights = alloc.values[i].clone();

        let allocs_change = !weights.iter().zip(prev_allocs.iter()).all(|(a, b)| a == b);

        let mut abs_w_sum = 0.0;
        for w in &weights {
            abs_w_sum += w.abs();
        }

        let mut h_t = vec![f64::NAN; n_cols];
        if i + 1 < n_rows && abs_w_sum != 0.0 {
            for j in 0..n_cols {
                let delever = weights[j] / abs_w_sum;
                let denom = open.values[i + 1][j] * rates.values[i][j];
                h_t[j] = delever / denom;
            }
        }

        let mut delta = vec![0.0; n_cols];
        for (j, delta_j) in delta.iter_mut().enumerate() {
            let close_open = close.values[i][j] - open.values[i][j];
            let price_diff = close.values[i][j] - close.values[i - 1][j];
            *delta_j = if prev_allocs_change { close_open } else { price_diff };
        }

        if prev_h.is_none() {
            prev_h = Some(h_t.iter().map(|v| v * prev_k).collect());
            out.push((open.index[i].clone(), prev_k));
            continue;
        }

        if prev_allocs_change {
            prev_h = Some(h_t.iter().map(|v| v * prev_k).collect());
        }

        let h_prev = prev_h.as_ref().ok_or(EtfTrickError::MissingPreviousHoldings)?;
        let mut k = prev_k;
        for j in 0..n_cols {
            k += h_prev[j] * rates.values[i][j] * (delta[j] + costs.values[i][j]);
        }
        out.push((open.index[i].clone(), k));

        prev_k = k;
        prev_allocs_change = allocs_change;
        prev_allocs = weights;
    }

    Ok(out)
}

/// One quote of a futures chain for [`get_futures_roll_series`].
#[derive(Clone, Debug)]
pub struct FuturesRollRow {
    /// Session date.
    pub date: NaiveDate,
    /// Opening price of `security`.
    pub open: f64,
    /// Closing price of `security`.
    pub close: f64,
    /// The contract this row quotes.
    pub security: String,
    /// The front contract on this date; rows where it differs from `security` are ignored.
    pub current_security: String,
}

/// Cumulative roll-gap series of a futures chain (AFML §2.4.3, Snippet 2.2).
///
/// Keeps the rows where `security == current_security`, sorts them by date, and at each
/// change of front contract takes the gap between the new contract's open and the old one's
/// previous close. Returns one value per kept row:
///
/// - `"absolute"`: cumulative sum of `open - previous close`; subtract it from raw prices;
/// - `"relative"`: cumulative product of `open / previous close`; divide raw prices by it.
///
/// With `roll_backward = true` the series is shifted so its last value is 0 (absolute) or 1
/// (relative): recent prices are left untouched and history is adjusted. An empty input, or
/// one with no front-contract rows, returns an empty vector. Mirrors mlfinlab's
/// `get_futures_roll_series`.
///
/// # Errors
///
/// [`EtfTrickError::UnknownRollMethod`] if `method` is not `"absolute"` or `"relative"` (only
/// checked when there are rows to roll).
///
/// ```
/// use chrono::NaiveDate;
/// use openquant::etf_trick::{get_futures_roll_series, FuturesRollRow};
///
/// # fn main() -> Result<(), openquant::etf_trick::EtfTrickError> {
/// let row = |day, open, close, contract: &str| FuturesRollRow {
///     date: NaiveDate::from_ymd_opt(2024, 1, day).unwrap(),
///     open,
///     close,
///     security: contract.to_string(),
///     current_security: contract.to_string(),
/// };
/// // The new contract opens 0.90 above the old one's last close.
/// let chain =
///     vec![row(3, 70.5, 71.2, "CLG4"), row(4, 71.4, 71.1, "CLG4"), row(5, 72.0, 72.9, "CLH4")];
/// let gaps = get_futures_roll_series(&chain, "absolute", true)?;
/// assert!((gaps[0] + 0.9).abs() < 1e-9 && (gaps[1] + 0.9).abs() < 1e-9);
/// assert_eq!(gaps[2], 0.0);
/// # Ok(())
/// # }
/// ```
pub fn get_futures_roll_series(
    rows: &[FuturesRollRow],
    method: &str,
    roll_backward: bool,
) -> Result<Vec<f64>, EtfTrickError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let mut filtered: Vec<FuturesRollRow> =
        rows.iter().filter(|r| r.security == r.current_security).cloned().collect();
    filtered.sort_by_key(|r| r.date);
    if filtered.is_empty() {
        return Ok(Vec::new());
    }

    // First index for each distinct current_security (roll dates).
    let mut roll_pos = Vec::new();
    let mut prev_sec: Option<&str> = None;
    for (i, r) in filtered.iter().enumerate() {
        let s = r.current_security.as_str();
        if prev_sec != Some(s) {
            roll_pos.push(i);
            prev_sec = Some(s);
        }
    }

    match method {
        "absolute" => {
            let mut gaps = vec![0.0; filtered.len()];
            for &pos in roll_pos.iter().skip(1) {
                gaps[pos] = filtered[pos].open - filtered[pos - 1].close;
            }
            let mut cum = 0.0;
            let mut out = Vec::with_capacity(gaps.len());
            for g in gaps {
                cum += g;
                out.push(cum);
            }
            if roll_backward {
                let last = *out.last().unwrap_or(&0.0);
                for v in &mut out {
                    *v -= last;
                }
            }
            Ok(out)
        }
        "relative" => {
            let mut gaps = vec![1.0; filtered.len()];
            for &pos in roll_pos.iter().skip(1) {
                gaps[pos] = filtered[pos].open / filtered[pos - 1].close;
            }
            let mut cum = 1.0;
            let mut out = Vec::with_capacity(gaps.len());
            for g in gaps {
                cum *= g;
                out.push(cum);
            }
            if roll_backward {
                let last = *out.last().unwrap_or(&1.0);
                for v in &mut out {
                    *v /= last;
                }
            }
            Ok(out)
        }
        _ => Err(EtfTrickError::UnknownRollMethod),
    }
}
