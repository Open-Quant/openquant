---
title: Benchmark Policy
description: Performance benchmarking and regression guardrails.
status: draft
banner:
  content: '<span class="doc-status doc-status--draft">Draft</span> This page is known to be incomplete. Treat its contents as provisional.'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

```bash
cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline
python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json
python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 25
```

Baseline policy:
- Treat >25% regression as a release blocker until investigated.
