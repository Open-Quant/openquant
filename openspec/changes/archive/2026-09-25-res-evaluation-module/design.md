## Context

`backtest_stats` (Rust `backtest_statistics`) implements PSR, DSR and MinTRL from moments;
`strategy_risk` implements the Chapter 15 failure probability. Both are bound to Python. The
missing piece is the research-facing layer: from a returns series, with a trial count that
persists. The package has no numpy dependency, so the layer is standard-library Python.

## Goals / Non-Goals

**Goals:**
- Compute PSR, DSR and MinTRL from returns using the existing Rust implementations.
- Persist every configuration tried so DSR deflates by the real number of trials.
- Meta-labeling overlay metrics and the strategy-failure probability in the same module.

**Non-Goals:**
- Clustering correlated trials into an effective N (AFML suggests ONC); the registry counts
  configurations as recorded.
- A database or lock-protected concurrent writes.
- Moving or deduplicating the Rust statistics helpers (issue #38).

## Decisions

- **Reuse the Rust formulas.** PSR, SR0 and MinTRL call `backtest_stats`; they were checked
  against an independent standard-library computation (`statistics.NormalDist`) and agree to
  1e-9, so no Rust change was needed. Moments (skewness, kurtosis) are computed in Python
  because no binding exposes them.
- **Registry DSR passes `[σ_SR, N]`** (`estimates_param=True`) rather than the list of Sharpe
  ratios, so N is the registry count by construction. σ_SR is the population standard
  deviation, matching the Rust list form.
- **JSON, not parquet.** A registry holds tens to thousands of rows; JSON is readable,
  diffable and needs no dependency. Writes go to a temporary file in the same directory,
  are fsynced and renamed over the target.
- **Re-recording replaces.** Identity is the SHA-256 of the configuration's canonical JSON.
  Re-running a notebook does not add trials; a genuinely different configuration must differ
  in its config.
- **MinTRL returns infinity** at or below the benchmark, instead of the Rust function's
  finite, meaningless value.

## Risks / Trade-offs

- [Correlated trials overstate N] → documented; users can record one representative per
  cluster.
- [Two processes writing at the same instant can lose a write] → documented; each write
  re-reads the file first, which covers sequential runs and separate sessions.
- [Mixing Sharpe ratios of different frequencies in one registry] → documented: every trial
  must be per-period at the same frequency.
