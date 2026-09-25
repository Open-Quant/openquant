# Vendored crates

## `pyo3-polars` (0.20.0, patched)

`vendor/pyo3-polars` is the crates.io release of
[`pyo3-polars` 0.20.0](https://crates.io/crates/pyo3-polars/0.20.0)
(upstream commit `9b69223c5fc51994e2802865c667cd62201b8397`, path
`pyo3-polars/` in `pola-rs/pyo3-polars`) with one functional change. The root
`Cargo.toml` swaps it in with:

```toml
[patch.crates-io]
pyo3-polars = { path = "vendor/pyo3-polars" }
```

It was added in `d811710` together with the polars `DataFrame` bindings in
`crates/pyopenquant/src/data.rs`, which are the only users.

### Who depends on it

Only `crates/pyopenquant`, the extension module inside the Python wheel.
The `openquant` crate does not depend on `pyo3-polars`, so it builds and
packages from crates.io without the patch. `pyopenquant` has
`publish = false`: it ships as a wheel built from this repository, never as a
crate. Cargo ignores `[patch]` for crates downloaded from a registry, so a
published `pyopenquant` would pick up the unpatched upstream crate.

### The delta

Everything except `src/types.rs` is byte-identical to the 0.20.0 crate
tarball, including the normalised `Cargo.toml`, `Cargo.toml.orig`,
`Cargo.lock` and `.cargo_vcs_info.json`. `.cargo-ok` is an extraction marker
written by Cargo.

In `src/types.rs`, `impl FromPyObject for PySeries` (Python to Rust) upstream
does:

```rust
let kwargs = PyDict::new(ob.py());
if let Ok(compat_level) = ob.call_method0("_newest_compat_level") {
    let compat_level = compat_level.extract().unwrap();
    let compat_level =
        CompatLevel::with_level(compat_level).unwrap_or(CompatLevel::newest());
    kwargs.set_item("compat_level", compat_level.get_level())?;
}
let arr = ob.call_method("to_arrow", (), Some(&kwargs))?;
```

The vendored copy does:

```rust
// Newer Python polars versions reject integer compat_level values.
// Fall back to the default to_arrow behavior for broad compatibility.
let arr = ob.call_method0("to_arrow")?;
```

The `pyo3::types::PyBytes` import, which only the `lazy` feature uses, is
behind `#[cfg(feature = "lazy")]`, so the default build has no unused-import
warning and the `lazy` feature still compiles.

### Why

Upstream passes a plain `int` as `compat_level` to Python
`Series.to_arrow`. Python polars 1.32.3 and later accept only a
`polars.CompatLevel` instance or `None`, and raise
``TypeError: `compat_level` has invalid type: 'int'``. With the unpatched
crate, every polars `DataFrame` passed into the extension fails with that
error (for example `openquant.data.clean_ohlcv`). `pyproject.toml` allows
`polars>=1.0,<2`, and the locked version in `uv.lock` is 1.38.1.

Calling `to_arrow()` with no argument uses the oldest compat level, which
polars 0.46 on the Rust side can always read. This can cost an extra copy
for string or binary columns (they are exported as large Arrow types rather
than views). It does not change values.

Rust to Python conversion (`IntoPyObject for PySeries`/`PyDataFrame`) is not
changed. It only reads `_newest_compat_level()` on the Rust side.

### Why it is not dropped yet

- No `pyo3-polars` 0.20.x release fixes this, and there is no upstream git
  revision that fixes it while still using polars 0.46 and pyo3 0.23.
- 0.21.0 (polars 0.48, pyo3 0.24) still has the same `to_arrow` call.
  Upstream replaced this path with the `_export` / `import_series` mechanism
  in later releases, up to 0.28.0 (polars 0.55, pyo3 0.29).
- Upgrading means moving the whole workspace (`openquant` and `pyopenquant`)
  from polars 0.46 to 0.55 and pyo3 from 0.23 to 0.29. That is a separate
  change with a much larger surface.

### How to remove it

Upgrade `polars`, `pyo3` and `pyo3-polars` together to a release whose
`FromPyObject for PySeries` does not pass an integer `compat_level`. Then
delete this directory and the `[patch.crates-io]` table, and run
`python/tests/test_data_module.py` against the current Python polars.

To check the delta again:

```bash
curl -L https://crates.io/api/v1/crates/pyo3-polars/0.20.0/download | tar xz -C /tmp
diff -r /tmp/pyo3-polars-0.20.0 vendor/pyo3-polars
```
