# Current execution diagnostics for exact replay

`runtime_context.collect_runtime_context()` returns JSON-safe diagnostic metadata for the current process: Python/platform, a fixed whitelist of numerical thread environment settings, NumPy/SciPy versions, NumPy build configuration, and actual BLAS threadpool information when optional `threadpoolctl` is installed. It adds no mandatory dependency. Unset environment settings are null; provider failures are explicit `status: unavailable` entries with exception type, not arbitrary exception text. An empty pool list means no supported pools were reported. It does not establish single-threaded execution.

NumPy's `show_config(mode="dicts")` supplies structured build metadata when supported. Unsupported older versions or missing optional providers remain unavailable. Provider output is captured to avoid corrupting a CLI JSON stream. Nonfinite metadata becomes null and unsupported object types receive explicit unavailable markers. The helper does not change thread settings or scientific arrays. Collection may import NumPy (already required by replay) and the optional threadpool inspector. It never imports SciPy just to obtain a version: an absent SciPy module is reported as unavailable/not_loaded. Standalone callers should note that importing NumPy may itself load a pool. Collect after scientific execution, when the libraries used by that execution are loaded. Because stdout/stderr redirection is process-global, invoke collection synchronously outside concurrent output-producing threads.

The replay CLI adds diagnostics after its selected operation. Direct callers can use:

```python
from experiments.runtime_context import collect_runtime_context

# After exact scientific comparison and saved-report identity calculation:
replay_result["execution_runtime"] = collect_runtime_context()
```

Keep this new diagnostic field outside scientific equality and outside the saved report's identity/hash. Do not require old reports to contain runtime metadata. The output explicitly says `current_execution_only` and `original_report_runtime: unknown_not_reconstructed`; it cannot recover the original report's settings. Even when metadata matches, cross-machine/build exactness is not guaranteed. Installation paths are diagnostic context, not portable identity keys.

This helper does not weaken exact comparison or reinterpret a mismatch as success. The retained six-run integration study demonstrated identical package versions and seed with different exact results under one versus two/five BLAS threads. The future-risk indicator counts remained unchanged there, while eight raw training-risk tie labels changed. Runtime context explains possible causes; it does not itself prove scientific equivalence.

NumPy documents backend-dependent controls in [global configuration](https://numpy.org/doc/2.3/reference/global_state.html) and runtime reporting in [show_runtime](https://numpy.org/doc/2.3/reference/generated/numpy.show_runtime.html). The helper records currently available metadata without claiming that environment strings establish actual pool sizes.

The replay CLI exposes this as the opt-in `--runtime-context` flag; its default output remains unchanged. When combined with `--check-identity`, the metadata describes the identity-check process only. Identity checking does not execute the scientific experiment, and reported loaded pools do not establish that those libraries performed its original calculation. More generally, a loaded pool is a runtime observation, not a trace of functions actually executed.
