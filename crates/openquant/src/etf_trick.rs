use std::collections::HashMap;
use std::path::Path;

use chrono::NaiveDate;
use csv::StringRecord;

#[derive(Clone, Debug)]
pub struct Table {
    pub index: Vec<String>,
    pub columns: Vec<String>,
    pub values: Vec<Vec<f64>>,
}

impl Table {
    pub fn from_csv(path: &Path) -> Result<Self, String> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let headers = rdr
            .headers()
            .map_err(|e| format!("failed to read headers {}: {e}", path.display()))?
            .clone();

        if headers.len() < 2 {
            return Err(format!(
                "csv {} must have at least index + 1 value column",
                path.display()
            ));
        }

        let columns = headers.iter().skip(1).map(ToString::to_string).collect::<Vec<_>>();
        let mut index = Vec::new();
        let mut values = Vec::new();

        for rec in rdr.records() {
            let record =
                rec.map_err(|e| format!("failed to read record {}: {e}", path.display()))?;
            let (idx, row) = parse_row(&record, path)?;
            index.push(idx);
            values.push(row);
        }

        Ok(Self { index, columns, values })
    }

    fn align_columns(&self, ordered_columns: &[String]) -> Result<Self, String> {
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
                    .ok_or_else(|| format!("missing column '{c}' in table"))?;
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

#[derive(Clone, Debug)]
pub struct EtfTrick {
    source: Source,
}

#[derive(Clone, Debug)]
enum Source {
    InMemory {
        open: Table,
        close: Table,
        alloc: Table,
        costs: Table,
        rates: Option<Table>,
    },
    Csv {
        open_path: String,
        close_path: String,
        alloc_path: String,
        costs_path: String,
        rates_path: Option<String>,
    },
}

impl EtfTrick {
    pub fn from_tables(
        open: Table,
        close: Table,
        alloc: Table,
        costs: Table,
        rates: Option<Table>,
    ) -> Result<Self, String> {
        validate_shapes(&open, &close, &alloc, &costs, rates.as_ref())?;
        Ok(Self { source: Source::InMemory { open, close, alloc, costs, rates } })
    }

    pub fn from_csv(
        open_path: &str,
        close_path: &str,
        alloc_path: &str,
        costs_path: &str,
        rates_path: Option<&str>,
    ) -> Result<Self, String> {
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

    pub fn get_etf_series(&self, batch_size: usize) -> Result<Vec<(String, f64)>, String> {
        match &self.source {
            Source::InMemory { open, close, alloc, costs, rates } => {
                compute_etf_series(open, close, alloc, costs, rates.as_ref())
            }
            Source::Csv { open_path, close_path, alloc_path, costs_path, rates_path } => {
                if batch_size < 3 {
                    return Err("Batch size should be >= 3".to_string());
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

    pub fn reset(&mut self) {}
}

fn parse_row(record: &StringRecord, path: &Path) -> Result<(String, Vec<f64>), String> {
    let idx = record
        .get(0)
        .ok_or_else(|| format!("missing index column in {}", path.display()))?
        .to_string();
    let mut row = Vec::with_capacity(record.len().saturating_sub(1));
    for cell in record.iter().skip(1) {
        let v = cell
            .parse::<f64>()
            .map_err(|e| format!("failed to parse float '{}' in {}: {e}", cell, path.display()))?;
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
) -> Result<(), String> {
    let pairs = [close, alloc, costs];
    for t in pairs {
        if open.index != t.index || open.values.len() != t.values.len() {
            return Err("DataFrames indices are different".to_string());
        }
        if open.columns.len() != t.columns.len() {
            return Err("DataFrames columns are different".to_string());
        }
    }
    if let Some(r) = rates {
        if open.index != r.index || open.values.len() != r.values.len() {
            return Err("DataFrames indices are different".to_string());
        }
        if open.columns.len() != r.columns.len() {
            return Err("DataFrames columns are different".to_string());
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
) -> Result<Vec<(String, f64)>, String> {
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
        for j in 0..n_cols {
            let close_open = close.values[i][j] - open.values[i][j];
            let price_diff = close.values[i][j] - close.values[i - 1][j];
            delta[j] = if prev_allocs_change { close_open } else { price_diff };
        }

        if prev_h.is_none() {
            prev_h = Some(h_t.iter().map(|v| v * prev_k).collect());
            out.push((open.index[i].clone(), prev_k));
            continue;
        }

        if prev_allocs_change {
            prev_h = Some(h_t.iter().map(|v| v * prev_k).collect());
        }

        let h_prev = prev_h.as_ref().ok_or_else(|| "missing previous h".to_string())?;
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

#[derive(Clone, Debug)]
pub struct FuturesRollRow {
    pub date: NaiveDate,
    pub open: f64,
    pub close: f64,
    pub security: String,
    pub current_security: String,
}

/// Generate rolling futures gap series.
/// Mirrors mlfinlab.multi_product.etf_trick.get_futures_roll_series.
pub fn get_futures_roll_series(
    rows: &[FuturesRollRow],
    method: &str,
    roll_backward: bool,
) -> Result<Vec<f64>, String> {
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
        _ => Err("The method must be either absolute or relative, Check spelling.".to_string()),
    }
}
