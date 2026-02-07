<p align="center">
  <img src="assets/banner_v2.svg" alt="OpenQuant-rs" width="100%" />
</p>

# OpenQuant-rs

OpenQuant-rs is a Rust-first library for modern machine learning methods in finance, inspired by and building on the ideas from *Advances in Financial Machine Learning* by Marcos Lopez de Prado.

## What this repo is
- A Rust implementation of core AFML techniques with a strict, testable baseline.
- A place to consolidate fixtures and baseline behavior before optimization.
- A stepping stone toward benchmarks and accelerated algorithms once parity is locked.

## Migration status
- Source of truth: `openquant-rs/tests/crosswalk.md`
- Project roadmap: `ROADMAP.md`

## Getting started
```bash
# Run all Rust tests
cargo test -p openquant

# Sync ROADMAP.md from the crosswalk
python scripts/sync_roadmap.py
```

## Structure
- `openquant-rs/crates/openquant/src/`: core library modules
- `openquant-rs/crates/openquant/tests/`: Rust test suite
- `openquant-rs/tests/fixtures/`: shared fixtures used for parity
- `openquant-rs/tests/crosswalk.md`: mapping of Python tests to Rust tests

## Principles
- Parity first: use crosswalk + shared fixtures to match Python behavior.
- Explicit tolerances: document any intentional deviations.
- Performance after parity: optimize only after tests are green and aligned.

## Contributing
If you are porting a module:
1) Add or update fixtures under `openquant-rs/tests/fixtures/`.
2) Implement Rust tests in `openquant-rs/crates/openquant/tests/`.
3) Update `openquant-rs/tests/crosswalk.md` with status and fixtures.
4) Run `python scripts/sync_roadmap.py` to keep the roadmap current.

## License
TBD
