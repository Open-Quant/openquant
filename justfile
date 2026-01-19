set shell := ["bash", "-cu"]

default := help

help:
    @just --list

fmt:
    cargo fmt

fmt-check:
    cargo fmt -- --check

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

check:
    cargo check --all-targets --all-features

test:
    cargo test --all-targets --all-features

lint: fmt-check clippy

bench:
    cargo bench --all-features
