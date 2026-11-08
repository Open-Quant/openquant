"""Best-effort diagnostics for the current execution, never scientific identity."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
import platform
import sys

THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPY_DISABLE_CPU_FEATURES",
    "OPENBLAS_CORETYPE",
)


def _json_value(value):
    """Keep output JSON-safe without evaluating arbitrary object representations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return {"status": "unavailable", "reason": "unsupported_metadata_type"}


def _optional(provider):
    """Optional providers must not write into replay's JSON output stream."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            value = _json_value(provider())
        # Validate here so malformed optional metadata cannot break final serialization.
        json.dumps(value, allow_nan=False)
        return {"status": "available", "value": value}
    except Exception as exc:
        return {"status": "unavailable", "reason": type(exc).__name__, "value": None}


def _loaded_version(name):
    module = sys.modules.get(name)
    if module is None:
        return {"status": "unavailable", "reason": "not_loaded", "value": None}
    return _optional(lambda: module.__version__)


def collect_runtime_context():
    """Describe this process only; do not infer a saved report's original runtime.

    This is diagnostic metadata. It must remain outside scientific output equality
    and the identity/hash of saved reports. threadpoolctl is optional. The caller
    should collect after scientific execution so its numerical libraries are loaded.
    """
    return {
        "schema_version": 1,
        "scope": "current_execution_only",
        "original_report_runtime": "unknown_not_reconstructed",
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "description": platform.platform(),
        },
        "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "numpy_version": _optional(lambda: importlib.import_module("numpy").__version__),
        "scipy_version": _loaded_version("scipy"),
        "numpy_build": _optional(
            lambda: importlib.import_module("numpy").show_config(mode="dicts")
        ),
        "threadpools": _optional(
            lambda: importlib.import_module("threadpoolctl").threadpool_info()
        ),
        "limitations": [
            "Only the current process is observed; the original report runtime is unknown.",
            "Unset environment variables do not imply single-threaded execution.",
            "An empty threadpool list means no supported pools were reported, not no threading.",
            "Matching context does not guarantee exact replay across hardware or builds.",
            "Build configuration may contain installation paths; it is not a portable identity.",
        ],
    }
