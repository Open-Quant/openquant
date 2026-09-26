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
//! - Allocations are de-levered by the sum of their absolute values. Holdings chosen at a
//!   rebalance on bar `t` use bar `t`'s allocation, `K_t` and FX, are sized at the **next**
//!   bar's open `o_{t+1}`, and earn from bar `t + 1` on; that bar earns open-to-close only.
//!   The first bar of the series counts as a rebalance.
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
//! // Six rows give four values: the series starts on the second row, and the last row lacks a
//! // next open.
//! assert_eq!(series.len(), 4);
//! assert_eq!(series[0], ("01-03".to_string(), 1.0));
//! // Bought at the 01-04 opens and held to the closes:
//! // 1 + 0.5 / 71.4 * (-0.30) + 0.5 / 2.49 * 0.05
//! assert!((series[1].1 - 1.007939).abs() < 1e-6);
//! # Ok(())
//! # }
//! ```

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
    /// Internal: holdings were not initialised before use. No longer returned; kept so that
    /// existing `match`es compile.
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
    /// Implements AFML §2.4.1: at a rebalance on bar `t` the holdings become
    /// `h_t = w_t K_t / (o_{t+1} fx_t sum|w_t|)`, and bar `t + 1` adds
    /// `sum h_t fx_{t+1} (delta_{t+1} + costs_{t+1})`, with `delta` the open-to-close change on
    /// the bar after a rebalance and the close-to-close change otherwise. Between rebalances
    /// the holdings are carried unchanged.
    ///
    /// The series starts on the second row with `K = 1`, and that row counts as a rebalance.
    /// The first row is not used, and the last row is dropped (sizing there needs a next open),
    /// so `n` rows give `n - 2` values; fewer than three rows give none. The row index matches
    /// mlfinlab's output; the values do not, because mlfinlab sizes the holdings one bar late
    /// (see the module page).
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

    // The series starts on the second row with K = 1 and stops one row before the end, where
    // sizing a position would need the next open.
    let n_rows = open.values.len();
    if n_rows < 3 {
        return Ok(Vec::new());
    }
    let n_cols = securities.len();

    // h_{i,t} = w_{i,t} K_t / (o_{i,t+1} phi_{i,t} sum_j |w_{j,t}|): the holdings set at the
    // close of a rebalance bar t, bought at the next open, which earn bar t + 1.
    let holdings = |t: usize, k: f64| -> Vec<f64> {
        let abs_w_sum: f64 = alloc.values[t].iter().map(|w| w.abs()).sum();
        (0..n_cols)
            .map(|j| {
                alloc.values[t][j] * k / (open.values[t + 1][j] * rates.values[t][j] * abs_w_sum)
            })
            .collect()
    };

    let start = 1;
    let mut k = 1.0f64;
    let mut out = vec![(open.index[start].clone(), k)];
    // The initial allocation is a rebalance: its holdings are bought at the next open.
    let mut h = holdings(start, k);
    let mut prev_rebalanced = true;

    for t in (start + 1)..(n_rows - 1) {
        // K_t = K_{t-1} + sum_i h_{i,t-1} phi_{i,t} (delta_{i,t} + d_{i,t}), where h_{t-1} was
        // sized from bar t-1's allocation, K and FX and bar t's open.
        for (j, h_j) in h.iter().enumerate() {
            let delta = if prev_rebalanced {
                close.values[t][j] - open.values[t][j]
            } else {
                close.values[t][j] - close.values[t - 1][j]
            };
            k += h_j * rates.values[t][j] * (delta + costs.values[t][j]);
        }
        out.push((open.index[t].clone(), k));

        let rebalanced = alloc.values[t] != alloc.values[t - 1];
        if rebalanced {
            h = holdings(t, k);
        }
        prev_rebalanced = rebalanced;
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
