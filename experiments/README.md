# Experiments Scaffold

This directory provides config-driven, reproducible research runs for OpenQuant.

## Run

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --out experiments/artifacts
```

Grid run (algo-wheel style sweep):

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --grid-config experiments/configs/futures_oil_grid.toml \
  --out experiments/artifacts
```

## Artifacts

Each run writes a deterministic folder:

- `run_manifest.json` (config hash, git sha, versions, seed, `dataset_hash`: the content hash of the data the run used, see `openquant.data.dataset_hash`)
- `metrics.parquet` (summary table)
- `events.parquet` (event stage output)
- `signals.parquet` (signal timeline)
- `weights.parquet` (portfolio weights)
- `backtest.parquet` (equity/returns/position)
- `equity_curve.svg` (equity line chart from `backtest.parquet`)
- `drawdown.svg` (drawdown from the running peak, `equity / peak - 1`)
- `decision.md` (promotion verdict and rationale)

The two SVGs are fixed-size and written without a plotting library, so the same config and
seed give byte-identical files.

Grid runs additionally include:
- `leaderboard.parquet` (ranked cross-run summary)
- one artifact subdirectory per grid entry with the same single-run files

## Notes

- Default config is futures-centric and oil-inclusive.
- Runner uses `openquant.research.run_flywheel_iteration` for notebook/script parity.

## Offline selection-validation lab

[The synthetic validation lab](VALIDATION_LAB.md) studies candidate-selection bias
using six controlled return families. It requires only NumPy and keeps training-only
selection separate from a deliberately contaminated holdout comparator:

```bash
python experiments/validation_lab.py --out /tmp/openquant-validation.json
python -m pytest python/tests/test_validation_lab.py -q
```

The JSON records all replications, input hashes, source/version provenance and timing.
These are synthetic diagnostics, not a trading strategy, calibrated confidence bound,
or promotion decision.

With the built OpenQuant extension, compare the existing PSR/DSR and trial-registry
diagnostics on the same controls:

```bash
.venv/bin/python experiments/compare_validation_diagnostics.py \
  --out /tmp/openquant-diagnostics.json
```

The [lab guide](VALIDATION_LAB.md#compare-the-existing-psrdsr-diagnostics) explains
score conventions, unavailable DSR values, registry audits and simulation intervals.

## Offline rolling risk-model comparison

[The risk-model utility lab](RISK_MODEL_LAB.md) compares existing minimum-variance,
HRP and baseline allocations on three controlled covariance families. Rolling fits
use only historical rows; the report separates realized risk, concentration, target
changes, drift-aware traded fractions and a declared toy cost model.

```bash
.venv/bin/python experiments/risk_model_lab.py --out /tmp/openquant-risk.json
```

All six methods are fixed before simulation. The lab reports independent-path and
paired Monte Carlo precision, unavailable paths, input hashes and native/source
provenance. It selects no winner and provides no investment guarantee.
