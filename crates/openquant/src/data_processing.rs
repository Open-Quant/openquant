use chrono::NaiveDateTime;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct OhlcvRow {
    pub timestamp: NaiveDateTime,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub adj_close: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignedOhlcvRow {
    pub timestamp: NaiveDateTime,
    pub symbol: String,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,
    pub adj_close: Option<f64>,
    pub is_missing_bar: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataQualityReport {
    pub row_count: usize,
    pub symbol_count: usize,
    pub duplicate_key_count: usize,
    pub gap_interval_count: usize,
    pub ts_min: Option<NaiveDateTime>,
    pub ts_max: Option<NaiveDateTime>,
    pub rows_removed_by_deduplication: usize,
}

fn sort_rows(rows: &mut [OhlcvRow]) {
    rows.sort_by(|a, b| {
        a.symbol
            .cmp(&b.symbol)
            .then_with(|| a.timestamp.cmp(&b.timestamp))
    });
}

fn dedupe_rows(rows: &[OhlcvRow], keep_last: bool) -> (Vec<OhlcvRow>, usize) {
    if rows.is_empty() {
        return (Vec::new(), 0);
    }

    let mut deduped = Vec::new();
    let mut i = 0usize;
    while i < rows.len() {
        let mut j = i + 1;
        while j < rows.len()
            && rows[j].symbol == rows[i].symbol
            && rows[j].timestamp == rows[i].timestamp
        {
            j += 1;
        }
        let chosen = if keep_last { &rows[j - 1] } else { &rows[i] };
        deduped.push(chosen.clone());
        i = j;
    }
    let removed = rows.len().saturating_sub(deduped.len());
    (deduped, removed)
}

pub fn quality_report(rows: &[OhlcvRow], rows_removed_by_deduplication: usize) -> DataQualityReport {
    let mut symbol_set = HashSet::new();
    let mut duplicate_key_count = 0usize;
    let mut key_counts: BTreeMap<(String, NaiveDateTime), usize> = BTreeMap::new();

    for row in rows {
        symbol_set.insert(row.symbol.clone());
        let key = (row.symbol.clone(), row.timestamp);
        *key_counts.entry(key).or_insert(0usize) += 1;
    }

    for count in key_counts.values() {
        if *count > 1 {
            duplicate_key_count += 1;
        }
    }

    let mut gap_interval_count = 0usize;
    let mut last_by_symbol: BTreeMap<String, NaiveDateTime> = BTreeMap::new();
    for row in rows {
        if let Some(prev) = last_by_symbol.get(&row.symbol) {
            if (row.timestamp - *prev).num_seconds() > 24 * 3600 {
                gap_interval_count += 1;
            }
        }
        last_by_symbol.insert(row.symbol.clone(), row.timestamp);
    }

    DataQualityReport {
        row_count: rows.len(),
        symbol_count: symbol_set.len(),
        duplicate_key_count,
        gap_interval_count,
        ts_min: rows.first().map(|r| r.timestamp),
        ts_max: rows.last().map(|r| r.timestamp),
        rows_removed_by_deduplication,
    }
}

pub fn clean_ohlcv_rows(rows: &[OhlcvRow], keep_last: bool) -> (Vec<OhlcvRow>, DataQualityReport) {
    let mut sorted = rows.to_vec();
    sort_rows(&mut sorted);
    let (deduped, removed) = dedupe_rows(&sorted, keep_last);
    let report = quality_report(&deduped, removed);
    (deduped, report)
}

pub fn align_calendar_rows(rows: &[OhlcvRow], interval_seconds: i64) -> Result<Vec<AlignedOhlcvRow>, String> {
    if interval_seconds <= 0 {
        return Err("interval_seconds must be > 0".to_string());
    }
    let (clean, _) = clean_ohlcv_rows(rows, true);
    if clean.is_empty() {
        return Ok(Vec::new());
    }

    let mut by_symbol: BTreeMap<String, Vec<OhlcvRow>> = BTreeMap::new();
    for row in clean {
        by_symbol.entry(row.symbol.clone()).or_default().push(row);
    }

    let mut out = Vec::new();
    for (symbol, rows_for_symbol) in by_symbol {
        if rows_for_symbol.is_empty() {
            continue;
        }
        let start = rows_for_symbol.first().expect("non-empty").timestamp;
        let end = rows_for_symbol.last().expect("non-empty").timestamp;

        let mut index: BTreeMap<NaiveDateTime, OhlcvRow> = BTreeMap::new();
        for row in rows_for_symbol {
            index.insert(row.timestamp, row);
        }

        let mut ts = start;
        while ts <= end {
            if let Some(row) = index.get(&ts) {
                out.push(AlignedOhlcvRow {
                    timestamp: ts,
                    symbol: symbol.clone(),
                    open: Some(row.open),
                    high: Some(row.high),
                    low: Some(row.low),
                    close: Some(row.close),
                    volume: Some(row.volume),
                    adj_close: Some(row.adj_close),
                    is_missing_bar: false,
                });
            } else {
                out.push(AlignedOhlcvRow {
                    timestamp: ts,
                    symbol: symbol.clone(),
                    open: None,
                    high: None,
                    low: None,
                    close: None,
                    volume: None,
                    adj_close: None,
                    is_missing_bar: true,
                });
            }
            ts += chrono::Duration::seconds(interval_seconds);
        }
    }
    Ok(out)
}
