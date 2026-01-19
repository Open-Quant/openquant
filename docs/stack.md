# Core crate stack (planned)

- `ndarray`: n-dimensional arrays and linear algebra; good BLAS/LAPACK integration and slicing APIs.
- `polars`: DataFrame operations and IO for CSV/Parquet fixtures when mirroring pandas workflows.
- `rayon`: Parallel iterators for filters/labeling hot paths.
- `serde` + `serde_json`/`serde_yaml`: Config and fixture serialization; interoperates with Python-exported fixtures.
- `statrs`: Statistical routines (distributions, moments) to mirror SciPy usage.
- `criterion`: Microbenchmarks for performance targets.

Notes:
- All crates will be added per-module as we port functionality to avoid unused deps.
- Prefer `no_std`-friendly choices only when it doesn’t conflict with performance goals; default is std.
- Keep features opt-in; consider `polars` as optional feature if binary size becomes a concern.
