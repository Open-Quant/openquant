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
//! - The matrix is dense: bars × labels bytes. The sequential bootstrap reads it once, then each
//!   draw costs order (nonzero cells + labels), so a full bootstrap is order
//!   labels × (nonzero cells + labels). Long, heavily overlapping spans push that towards
//!   bars × labels²: bootstrap within blocks, or draw fewer samples, for large label sets.
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

use crate::util::InputError;
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::Rng;

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
    validate_spans(label_endtime)?;
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

/// Rejects the first span with `start > end`.
fn validate_spans(label_endtime: &[(usize, usize)]) -> Result<(), InputError> {
    match label_endtime.iter().find(|(s, e)| s > e) {
        Some((_, e)) => Err(InputError::OutOfRange {
            name: "label_endtime",
            value: *e as f64,
            expected: "an end at or after the label's start",
        }),
        None => Ok(()),
    }
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
    // One row-major pass: each row's concurrency is summed once (not once per alive cell),
    // and each label's sums still accumulate over the rows in order, so the result is
    // bit-for-bit the column-by-column formula.
    let mut numer = vec![0.0; cols];
    let mut denom = vec![0.0; cols];
    for row in ind_mat {
        let mut inv_conc = None;
        for ((&v, n), d) in row.iter().zip(numer.iter_mut()).zip(denom.iter_mut()) {
            if v == 1 {
                *n += *inv_conc.get_or_insert_with(|| 1.0 / row_concurrency(row));
                *d += 1.0;
            }
        }
    }
    let mut uniq_sum = 0.0;
    let mut count = 0;
    for (n, d) in numer.iter().zip(&denom) {
        if *d > 0.0 {
            uniq_sum += n / d;
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
    // Each row's concurrency is summed once, not once per (label, bar) cell.
    let mut out = vec![vec![0.0; ind_mat.len()]; cols];
    for (t, row) in ind_mat.iter().enumerate() {
        let mut inv_conc = None;
        for (vals, &v) in out.iter_mut().zip(row) {
            if v == 1 {
                vals[t] = *inv_conc.get_or_insert_with(|| 1.0 / row_concurrency(row));
            }
        }
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
    // Row-major, so the matrix is read contiguously; each label's running mean still
    // advances over the bars in order, giving the same result bit for bit.
    let mut avg_unique = vec![0.0; cols];
    let mut n = vec![0.0; cols];
    for (row, &prev) in ind_mat.iter().zip(prev_concurrency) {
        for ((&cell, avg), cnt) in row.iter().zip(avg_unique.iter_mut()).zip(n.iter_mut()) {
            if cell > 0 {
                running_uniqueness_step(avg, cnt, f64::from(cell), prev);
            }
        }
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
    seq_bootstrap_with_rng(ind_mat, sample_length, warmup_samples, &mut rand::rng())
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
    // A label spans a few bars, so the matrix is mostly zeros: each draw walks every label's
    // nonzero cells in bar order instead of every cell. That is `bootstrap_loop_run`'s
    // arithmetic in the same order, so the weights, and hence the draws, are unchanged.
    let columns = SparseColumns::new(ind_mat, n_labels);
    let mut avg_unique = vec![0.0; n_labels];

    while phi.len() < target_len {
        let choice = match warm.pop() {
            Some(w) => w,
            None => {
                columns.average_uniqueness(&prev_conc, &mut avg_unique);
                let sum: f64 = avg_unique.iter().sum();
                let prob_iter = avg_unique.iter().map(|p| if sum > 0.0 { *p / sum } else { 1.0 });
                // Weights are non-negative, finite and not all zero by construction.
                let dist = WeightedIndex::new(prob_iter).expect("valid sampling weights");
                dist.sample(rng)
            }
        };
        phi.push(choice);
        // Adding a zero cell leaves `prev_conc` unchanged, so only the nonzero cells matter.
        for (&bar, &val) in columns.column(choice) {
            prev_conc[bar] += val;
        }
    }
    Ok(phi)
}

/// Concurrency `c_t` of one indicator-matrix row: the sum of its cells.
fn row_concurrency(row: &[u8]) -> f64 {
    row.iter().map(|v| *v as f64).sum::<f64>()
}

/// One step of [`bootstrap_loop_run`]'s running mean of `val / (val + prev)` over a label's
/// bars.
#[inline]
fn running_uniqueness_step(avg: &mut f64, n: &mut f64, val: f64, prev: f64) {
    let new_el = val / (val + prev);
    *avg = (*avg * *n + new_el) / (*n + 1.0);
    *n += 1.0;
}

/// The nonzero cells of an indicator matrix, label by label (compressed sparse columns),
/// with bars ascending within each label.
struct SparseColumns {
    /// Label `j`'s cells are at `starts[j]..starts[j + 1]` in `bars` and `vals`.
    starts: Vec<usize>,
    bars: Vec<usize>,
    vals: Vec<f64>,
}

impl SparseColumns {
    /// `ind_mat` must have rows of length `n_labels`.
    fn new(ind_mat: &[Vec<u8>], n_labels: usize) -> Self {
        let mut starts = vec![0usize; n_labels + 1];
        for row in ind_mat {
            for (count, &v) in starts[1..].iter_mut().zip(row) {
                if v > 0 {
                    *count += 1;
                }
            }
        }
        for j in 0..n_labels {
            starts[j + 1] += starts[j];
        }
        let nnz = starts[n_labels];
        let mut bars = vec![0usize; nnz];
        let mut vals = vec![0.0; nnz];
        let mut next = starts[..n_labels].to_vec();
        for (bar, row) in ind_mat.iter().enumerate() {
            for (pos, &v) in next.iter_mut().zip(row) {
                if v > 0 {
                    bars[*pos] = bar;
                    vals[*pos] = f64::from(v);
                    *pos += 1;
                }
            }
        }
        Self { starts, bars, vals }
    }

    /// Label `j`'s `(bar, value)` cells, bars ascending.
    fn column(&self, j: usize) -> impl Iterator<Item = (&usize, &f64)> {
        let range = self.starts[j]..self.starts[j + 1];
        self.bars[range.clone()].iter().zip(&self.vals[range])
    }

    /// [`bootstrap_loop_run`] on the matrix these columns came from, written into `out`.
    fn average_uniqueness(&self, prev_concurrency: &[f64], out: &mut [f64]) {
        for (j, slot) in out.iter_mut().enumerate() {
            let mut avg = 0.0;
            let mut n = 0.0;
            for (&bar, &val) in self.column(j) {
                running_uniqueness_step(&mut avg, &mut n, val, prev_concurrency[bar]);
            }
            *slot = avg;
        }
    }
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
    validate_spans(samples_info)?;
    // With no bars the indicator matrix has no rows, hence no label columns either.
    if price_bars_len == 0 {
        return Ok(Vec::new());
    }
    // Straight from the spans, without the bars × labels indicator matrix and its per-label
    // uniqueness rows: `c_t` is the same count, and each label sums `1 / c_t` over its bars
    // in ascending order, as the matrix route did, so the result is identical.
    let counts = num_concurrent_events(price_bars_len, samples_info, &[]);
    Ok(samples_info
        .iter()
        .map(|&(start, end)| {
            let span = counts.get(start..=end.min(price_bars_len - 1)).unwrap_or(&[]);
            if span.is_empty() {
                return 0.0;
            }
            let sum = span.iter().fold(0.0, |acc, &c| acc + 1.0 / c as f64);
            sum / span.len() as f64
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
