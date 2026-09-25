//! Label concurrency, average uniqueness and the sequential bootstrap (AFML chapter 4).
//!
//! Two triple-barrier labels whose spans overlap are statements about some of the same
//! returns, so they are not independent observations (AFML §4.2). This module measures that
//! overlap and draws a bootstrap sample around it:
//!
//! - [`get_ind_matrix`] builds the bar-by-label indicator matrix `1_{t,i}` (AFML Snippet 4.3).
//! - [`num_concurrent_events`] counts the labels alive at each bar, `c_t` (Snippet 4.1).
//! - [`get_ind_mat_label_uniqueness`], [`get_ind_mat_average_uniqueness`] and
//!   [`get_av_uniqueness_from_triple_barrier`] give per-bar uniqueness `u_{t,i} = 1_{t,i} / c_t`
//!   and its average over each label's lifespan (Snippets 4.2 and 4.4).
//! - [`seq_bootstrap`] / [`seq_bootstrap_with_rng`] draw labels one at a time, each with
//!   probability proportional to the average uniqueness it would have given the draws so far
//!   (Snippet 4.5), with [`bootstrap_loop_run`] as the single step.
//!
//! [`crate::sample_weights`] turns the same concurrency counts into training weights.
//!
//! Conventions:
//!
//! - Everything works in **bar positions**, not timestamps. A label is a pair
//!   `(start, end)` of **inclusive** bar indices; convert event and barrier-touch times to
//!   positions in the same bar series first.
//! - An indicator matrix is `Vec<Vec<u8>>` with **one row per bar and one column per label**,
//!   holding 0 or 1. Functions that take one reject ragged rows.
//! - A span that runs past the last bar is truncated silently, not rejected.
//! - The matrix is dense: bars × labels bytes, and a full sequential bootstrap rescans all of it
//!   for every draw (order bars × labels² work). Bootstrap within blocks, or draw fewer samples,
//!   for large label sets.
//!
//! The example is AFML §4.5.3's worked case: three labels over six bars.
//!
//! ```
//! use openquant::sampling::{
//!     bootstrap_loop_run, get_av_uniqueness_from_triple_barrier, get_ind_matrix,
//! };
//!
//! let spans = vec![(0, 2), (2, 3), (4, 5)];
//! let bars: Vec<usize> = (0..6).collect();
//! let ind = get_ind_matrix(&spans, &bars).unwrap();
//! assert_eq!(ind[2], vec![1, 1, 0]); // labels 0 and 1 share bar 2
//!
//! let uniqueness = get_av_uniqueness_from_triple_barrier(&spans, bars.len()).unwrap();
//! assert!((uniqueness[0] - 5.0 / 6.0).abs() < 1e-12);
//! assert!((uniqueness[1] - 0.75).abs() < 1e-12);
//! assert_eq!(uniqueness[2], 1.0);
//!
//! // Second-draw probabilities after label 1 was drawn: the book's {5/14, 3/14, 6/14}.
//! let concurrency: Vec<f64> = ind.iter().map(|row| f64::from(row[1])).collect();
//! let u = bootstrap_loop_run(&ind, &concurrency).unwrap();
//! let total: f64 = u.iter().sum();
//! assert!((u[0] / total - 5.0 / 14.0).abs() < 1e-12);
//! assert!((u[1] / total - 3.0 / 14.0).abs() < 1e-12);
//! assert!((u[2] / total - 6.0 / 14.0).abs() < 1e-12);
//! ```
#![deny(missing_docs)]

use crate::util::InputError;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{thread_rng, Rng};

/// Builds the indicator matrix of which labels span which bars (AFML Snippet 4.3).
///
/// `label_endtime[i] = (start, end)` is label `i`'s inclusive span in bar positions and
/// `bar_index` lists the bars to report, in the order the rows should appear. Entry `[r][i]` is
/// 1 when `start <= bar_index[r] <= end` and 0 otherwise, so the result has
/// `bar_index.len()` rows and `label_endtime.len()` columns. A span reaching past the bars
/// listed is truncated silently; `bar_index` does not have to be contiguous.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if any label has `start > end`.
///
/// ```
/// use openquant::sampling::get_ind_matrix;
///
/// let ind = get_ind_matrix(&[(0, 1), (1, 9)], &[0, 1, 2]).unwrap();
/// assert_eq!(ind, vec![vec![1, 0], vec![1, 1], vec![0, 1]]);
/// assert!(get_ind_matrix(&[(3, 2)], &[0, 1]).is_err());
/// ```
pub fn get_ind_matrix(
    label_endtime: &[(usize, usize)],
    bar_index: &[usize],
) -> Result<Vec<Vec<u8>>, InputError> {
    if let Some((_, e)) = label_endtime.iter().find(|(s, e)| s > e) {
        return Err(InputError::OutOfRange {
            name: "label_endtime",
            value: *e as f64,
            expected: "an end at or after the label's start",
        });
    }
    let mut ind = vec![vec![0u8; label_endtime.len()]; bar_index.len()];
    for (col, (start, end)) in label_endtime.iter().enumerate() {
        for (row_idx, bar) in bar_index.iter().enumerate() {
            if *bar >= *start && *bar <= *end {
                ind[row_idx][col] = 1;
            }
        }
    }
    Ok(ind)
}

/// Number of label columns, after checking that every row has that many.
fn label_count(ind_mat: &[Vec<u8>]) -> Result<usize, InputError> {
    let cols = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    match ind_mat.iter().find(|row| row.len() != cols) {
        Some(row) => {
            Err(InputError::LengthMismatch { name: "ind_mat row", len: row.len(), expected: cols })
        }
        None => Ok(cols),
    }
}

/// Returns the mean, over labels, of each label's average uniqueness (AFML Snippet 4.4 applied
/// to a whole sample).
///
/// For each label `i` the average uniqueness is `ū_i = Σ_t u_{t,i} / Σ_t 1_{t,i}`, with
/// `u_{t,i} = 1_{t,i} / c_t` and `c_t` the row sum of `ind_mat` at bar `t`. The result is the
/// plain mean of `ū_i` over the labels that span at least one bar; labels with an all-zero
/// column are left out. An empty matrix, or one with no label spanning any bar, gives `0.0`.
///
/// Only cells equal to exactly 1 count as "label alive", but `c_t` sums the raw cell values,
/// so pass a 0/1 matrix. Repeated columns (a bootstrap sample restricted to its drawn columns)
/// are allowed and lower each other's uniqueness, which is how the module docs page compares
/// bootstrap samples.
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if the rows of `ind_mat` do not all have the same length.
///
/// ```
/// use openquant::sampling::{get_ind_mat_average_uniqueness, get_ind_matrix};
///
/// let ind = get_ind_matrix(&[(0, 2), (2, 3), (4, 5)], &[0, 1, 2, 3, 4, 5]).unwrap();
/// let avg = get_ind_mat_average_uniqueness(&ind).unwrap();
/// // Mean of 5/6, 3/4 and 1.
/// assert!((avg - (5.0 / 6.0 + 0.75 + 1.0) / 3.0).abs() < 1e-12);
/// ```
pub fn get_ind_mat_average_uniqueness(ind_mat: &[Vec<u8>]) -> Result<f64, InputError> {
    let cols = label_count(ind_mat)?;
    if cols == 0 {
        return Ok(0.0);
    }
    let mut uniq_sum = 0.0;
    let mut count = 0;
    for col in 0..cols {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for row in ind_mat {
            if row[col] == 1 {
                let conc = row.iter().map(|v| *v as f64).sum::<f64>();
                numer += 1.0 / conc;
                denom += 1.0;
            }
        }
        if denom > 0.0 {
            uniq_sum += numer / denom;
            count += 1;
        }
    }
    Ok(if count > 0 { uniq_sum / count as f64 } else { 0.0 })
}

/// Returns the per-bar uniqueness `u_{t,i} = 1_{t,i} / c_t` of every label (AFML §4.4).
///
/// The result is indexed **label first**: `result[i][t]` is label `i`'s uniqueness at bar
/// (row) `t`, and 0 at bars the label does not span. `c_t` is the row sum of `ind_mat`. An
/// empty matrix gives an empty result.
///
/// # Errors
///
/// [`InputError::LengthMismatch`] if the rows of `ind_mat` do not all have the same length.
///
/// ```
/// use openquant::sampling::get_ind_mat_label_uniqueness;
///
/// let ind = vec![vec![1, 0], vec![1, 1], vec![0, 1]];
/// let u = get_ind_mat_label_uniqueness(&ind).unwrap();
/// assert_eq!(u, vec![vec![1.0, 0.5, 0.0], vec![0.0, 0.5, 1.0]]);
/// ```
pub fn get_ind_mat_label_uniqueness(ind_mat: &[Vec<u8>]) -> Result<Vec<Vec<f64>>, InputError> {
    let cols = label_count(ind_mat)?;
    let mut out = vec![Vec::new(); cols];
    for col in 0..cols {
        let mut vals = Vec::new();
        for row in ind_mat {
            if row[col] == 1 {
                let conc = row.iter().map(|v| *v as f64).sum::<f64>();
                vals.push(1.0 / conc);
            } else {
                vals.push(0.0);
            }
        }
        out[col] = vals;
    }
    Ok(out)
}

/// One step of the sequential bootstrap: the average uniqueness each label would have if it
/// were drawn next (the inner loop of AFML Snippet 4.5).
///
/// `prev_concurrency[t]` is how many times bar `t` is already covered by the labels drawn so
/// far (one entry per row of `ind_mat`). For each label `j` the result is the mean, over the
/// bars `t` it spans, of `1 / (1 + prev_concurrency[t])` (strictly `v / (v + prev_concurrency[t])`
/// with `v` the cell value, which is the same for a 0/1 matrix). A label that spans no bar gets
/// 0. Normalising the result to sum to one gives the draw probabilities `δ_j`.
///
/// # Errors
///
/// - [`InputError::LengthMismatch`] if the rows of `ind_mat` do not all have the same length.
/// - [`InputError::LengthMismatch`] if `prev_concurrency` does not have one entry per row of
///   `ind_mat`.
///
/// ```
/// use openquant::sampling::bootstrap_loop_run;
///
/// let ind = vec![vec![1, 0], vec![1, 1], vec![0, 1]];
/// // Nothing drawn yet: each label's average uniqueness is 1.
/// assert_eq!(bootstrap_loop_run(&ind, &[0.0, 0.0, 0.0]).unwrap(), vec![1.0, 1.0]);
/// // After drawing label 0: label 0 would score 1/2 everywhere, label 1 (1/2 + 1) / 2.
/// assert_eq!(bootstrap_loop_run(&ind, &[1.0, 1.0, 0.0]).unwrap(), vec![0.5, 0.75]);
/// ```
pub fn bootstrap_loop_run(
    ind_mat: &[Vec<u8>],
    prev_concurrency: &[f64],
) -> Result<Vec<f64>, InputError> {
    let cols = label_count(ind_mat)?;
    if prev_concurrency.len() != ind_mat.len() {
        return Err(InputError::LengthMismatch {
            name: "prev_concurrency",
            len: prev_concurrency.len(),
            expected: ind_mat.len(),
        });
    }
    let mut avg_unique = vec![0.0; cols];
    for i in 0..cols {
        let mut prev_avg = 0.0;
        let mut n = 0.0;
        for (j, row) in ind_mat.iter().enumerate() {
            let val = row[i] as f64;
            if val > 0.0 {
                let new_el = val / (val + prev_concurrency[j]);
                let avg = (prev_avg * n + new_el) / (n + 1.0);
                n += 1.0;
                prev_avg = avg;
            }
        }
        avg_unique[i] = prev_avg;
    }
    Ok(avg_unique)
}

/// Sequential bootstrap (AFML Snippet 4.5): label indices drawn one at a time, each with
/// probability proportional to its average uniqueness given the draws so far.
///
/// Draws from the thread-local generator, so repeated calls differ. Use
/// [`seq_bootstrap_with_rng`] for a reproducible sample; the arguments and result are the same
/// as there.
///
/// # Errors
///
/// As [`seq_bootstrap_with_rng`].
pub fn seq_bootstrap(
    ind_mat: &[Vec<u8>],
    sample_length: Option<usize>,
    warmup_samples: Option<Vec<usize>>,
) -> Result<Vec<usize>, InputError> {
    seq_bootstrap_with_rng(ind_mat, sample_length, warmup_samples, &mut thread_rng())
}

/// [`seq_bootstrap`] drawing from the supplied generator; a seeded generator gives a
/// reproducible sample.
///
/// `ind_mat` is a bar-by-label 0/1 matrix (see [`get_ind_matrix`]). Returns the drawn label
/// (column) indices in draw order; repeats are possible, only less likely than under a
/// uniform bootstrap. `sample_length` defaults to the number of labels. `warmup_samples`
/// forces the first draws and is consumed from the **end** of the list (so `[2, 0]` draws
/// label 0 first); every later draw is from the uniqueness-weighted distribution computed by
/// [`bootstrap_loop_run`]. If no label spans any bar, the later draws are uniform.
///
/// A requested length of zero returns an empty sample before any other check, so an empty
/// matrix with the default `sample_length` (zero labels) gives `Ok(vec![])`.
///
/// # Errors
///
/// - [`InputError::LengthMismatch`] if the rows of `ind_mat` do not all have the same length.
/// - [`InputError::TooShort`] if `ind_mat` has no labels and a non-zero `sample_length` was
///   requested.
/// - [`InputError::OutOfRange`] if a `warmup_samples` index is not a valid label index.
///
/// ```
/// use openquant::sampling::seq_bootstrap_with_rng;
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let ind = vec![vec![1, 0, 0], vec![1, 1, 0], vec![0, 1, 0], vec![0, 0, 1]];
/// let mut rng = StdRng::seed_from_u64(7);
///
/// // Warm-up draws are taken from the end of the list.
/// let drawn = seq_bootstrap_with_rng(&ind, Some(2), Some(vec![2, 0]), &mut rng).unwrap();
/// assert_eq!(drawn, vec![0, 2]);
///
/// // The same seed gives the same sample.
/// let a = seq_bootstrap_with_rng(&ind, None, None, &mut StdRng::seed_from_u64(1)).unwrap();
/// let b = seq_bootstrap_with_rng(&ind, None, None, &mut StdRng::seed_from_u64(1)).unwrap();
/// assert_eq!(a, b);
/// assert_eq!(a.len(), 3);
/// ```
pub fn seq_bootstrap_with_rng<R: Rng + ?Sized>(
    ind_mat: &[Vec<u8>],
    sample_length: Option<usize>,
    warmup_samples: Option<Vec<usize>>,
    rng: &mut R,
) -> Result<Vec<usize>, InputError> {
    let n_labels = label_count(ind_mat)?;
    let target_len = sample_length.unwrap_or(n_labels);
    if target_len == 0 {
        return Ok(Vec::new());
    }
    if n_labels == 0 {
        return Err(InputError::TooShort { name: "ind_mat", len: 0, min: 1 });
    }
    let mut phi: Vec<usize> = Vec::with_capacity(target_len);
    let mut warm = warmup_samples.unwrap_or_default();
    if let Some(&bad) = warm.iter().find(|&&w| w >= n_labels) {
        return Err(InputError::OutOfRange {
            name: "warmup_samples",
            value: bad as f64,
            expected: "a label index below the number of labels",
        });
    }
    let mut prev_conc = vec![0.0; ind_mat.len()];

    while phi.len() < target_len {
        let choice = match warm.pop() {
            Some(w) => w,
            None => {
                let avg_unique = bootstrap_loop_run(ind_mat, &prev_conc)?;
                let sum: f64 = avg_unique.iter().sum();
                let prob_iter = avg_unique.iter().map(|p| if sum > 0.0 { *p / sum } else { 1.0 });
                // Weights are non-negative, finite and not all zero by construction.
                let dist = WeightedIndex::new(prob_iter).expect("valid sampling weights");
                dist.sample(rng)
            }
        };
        phi.push(choice);
        for (i, row) in ind_mat.iter().enumerate() {
            prev_conc[i] += row[choice] as f64;
        }
    }
    Ok(phi)
}

/// Returns each label's average uniqueness over its lifespan, from its span (AFML
/// Snippet 4.4).
///
/// `samples_info[i] = (start, end)` is label `i`'s inclusive span in bar positions, and the
/// bars are `0..price_bars_len`. The result has one entry per label:
/// `ū_i = Σ_t u_{t,i} / Σ_t 1_{t,i}`, `1.0` for a label that overlaps nothing, and `0.0` for a
/// label that spans no bar in range. Spans running past `price_bars_len - 1` are truncated.
///
/// # Errors
///
/// [`InputError::OutOfRange`] if any label has `start > end`.
///
/// ```
/// use openquant::sampling::get_av_uniqueness_from_triple_barrier;
///
/// let u = get_av_uniqueness_from_triple_barrier(&[(0, 1), (1, 2)], 3).unwrap();
/// assert_eq!(u, vec![0.75, 0.75]);
/// ```
pub fn get_av_uniqueness_from_triple_barrier(
    samples_info: &[(usize, usize)],
    price_bars_len: usize,
) -> Result<Vec<f64>, InputError> {
    let bars: Vec<usize> = (0..price_bars_len).collect();
    let ind = get_ind_matrix(samples_info, &bars)?;
    let uniq = get_ind_mat_label_uniqueness(&ind)?;
    Ok(uniq
        .iter()
        .map(|u| {
            let sum: f64 = u.iter().filter(|v| **v > 0.0).sum();
            let cnt = u.iter().filter(|v| **v > 0.0).count() as f64;
            if cnt > 0.0 {
                sum / cnt
            } else {
                0.0
            }
        })
        .collect())
}

/// Counts the labels alive at each bar, `c_t` (AFML Snippet 4.1).
///
/// `t1[i] = (start, end)` is label `i`'s inclusive span in bar positions and the result has
/// `price_index_len` entries. Spans are truncated at the last bar, and a span with
/// `start > end` is skipped silently (where [`get_ind_matrix`] rejects it). `_t_events` is
/// ignored; it is kept for signature compatibility with the snippet, so pass an empty slice.
///
/// ```
/// use openquant::sampling::num_concurrent_events;
///
/// let counts = num_concurrent_events(6, &[(0, 2), (2, 3), (4, 9)], &[]);
/// assert_eq!(counts, vec![1, 1, 2, 1, 1, 1]);
/// ```
pub fn num_concurrent_events(
    price_index_len: usize,
    t1: &[(usize, usize)],
    _t_events: &[usize],
) -> Vec<usize> {
    if price_index_len == 0 {
        return Vec::new();
    }
    let mut counts = vec![0usize; price_index_len];
    for &(start, end) in t1 {
        if start > end {
            continue;
        }
        let end_idx = end.min(price_index_len - 1);
        for count in counts.iter_mut().take(end_idx + 1).skip(start) {
            *count += 1;
        }
    }
    counts
}
