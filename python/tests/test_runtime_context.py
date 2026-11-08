import json
from types import SimpleNamespace

import pytest

from experiments import runtime_context as runtime


def providers(monkeypatch, pools=None):
    modules = {
        "numpy": SimpleNamespace(
            __version__="test-numpy", show_config=lambda **kw: {"BLAS": "test"}
        ),
        "scipy": SimpleNamespace(__version__="test-scipy"),
        "threadpoolctl": SimpleNamespace(threadpool_info=lambda: [] if pools is None else pools),
    }
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setitem(runtime.sys.modules, "scipy", modules["scipy"])
    return modules


def test_current_context_and_json(monkeypatch):
    providers(monkeypatch, [{"num_threads": 2, "internal_api": "openblas"}])
    result = runtime.collect_runtime_context()
    assert result["scope"] == "current_execution_only"
    assert result["original_report_runtime"] == "unknown_not_reconstructed"
    assert result["threadpools"]["value"][0]["num_threads"] == 2
    assert result["numpy_build"]["value"] == {"BLAS": "test"}
    json.dumps(result, allow_nan=False)


def test_environment_whitelist_and_unset(monkeypatch):
    providers(monkeypatch)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-appear")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    result = runtime.collect_runtime_context()
    assert result["environment"]["OPENBLAS_NUM_THREADS"] == "3"
    assert result["environment"]["OMP_NUM_THREADS"] is None
    assert "must-not-appear" not in json.dumps(result)


def test_missing_optional_dependency(monkeypatch):
    modules = providers(monkeypatch)

    def load(name):
        if name == "threadpoolctl":
            raise ImportError("optional missing")
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", load)
    result = runtime.collect_runtime_context()
    assert result["threadpools"] == {
        "status": "unavailable",
        "reason": "ImportError",
        "value": None,
    }
    assert result["numpy_build"]["status"] == "available"


@pytest.mark.parametrize("error", [TypeError, RuntimeError, ValueError])
def test_provider_failure_is_explicit_and_quiet(monkeypatch, capsys, error):
    modules = providers(monkeypatch)

    def bad(**kwargs):
        print("unexpected provider output")
        raise error("do not leak arbitrary provider text")

    modules["numpy"].show_config = bad
    result = runtime.collect_runtime_context()
    assert result["numpy_build"]["reason"] == error.__name__
    assert capsys.readouterr().out == ""
    assert "do not leak" not in json.dumps(result)


def test_nonfinite_and_unknown_metadata(monkeypatch):
    providers(monkeypatch, [{"nan": float("nan"), "object": object()}])
    result = runtime.collect_runtime_context()
    assert result["threadpools"]["value"][0]["nan"] is None
    assert result["threadpools"]["value"][0]["object"]["status"] == "unavailable"
    json.dumps(result, allow_nan=False)


def test_empty_pools_does_not_claim_single_thread(monkeypatch):
    providers(monkeypatch)
    result = runtime.collect_runtime_context()
    assert result["threadpools"] == {"status": "available", "value": []}
    assert any("not no threading" in note for note in result["limitations"])


def test_scipy_is_never_imported_for_metadata(monkeypatch):
    modules = providers(monkeypatch)
    monkeypatch.delitem(runtime.sys.modules, "scipy", raising=False)
    imported = []

    def load(name):
        imported.append(name)
        assert name != "scipy"
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", load)
    result = runtime.collect_runtime_context()
    assert "scipy" not in imported
    assert result["scipy_version"] == {
        "status": "unavailable",
        "reason": "not_loaded",
        "value": None,
    }
