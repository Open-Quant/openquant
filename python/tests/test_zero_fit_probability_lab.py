import numpy as np
import pytest

pytest.importorskip("scipy", reason="optional zero-fit research dependency")

from experiments import zero_fit_probability_lab as lab


def test_exact_formula():
    assert lab.wendel(8, 1) == 127 / 128
    assert lab.wendel(8, 3) == 99 / 128
    assert lab.wendel(8, 7) == 1 / 128
    assert lab.wendel(8, 8) == 0


@pytest.mark.parametrize(
    "a,status",
    [
        ([[1, -1]], "member"),
        ([[1, 2]], "separated"),
        ([[0, 1]], "member"),
        ([[1, 0], [0, 1]], "separated"),
        ([[1, -1, 0], [0, 0, 1]], "member"),
    ],
)
def test_classification(a, status):
    assert lab.classify(a)["status"] == status


def test_positive_column_scale_invariance():
    for a in (np.array([[1.0, -1, 0], [0, 0, 1]]), np.eye(3)):
        assert lab.classify(a)["status"] == lab.classify(a * np.array([1e-100, 3, 1e100]))["status"]


@pytest.mark.parametrize("a", [[], [1, 2], [[1j]], [[np.nan]], np.empty((0, 2))])
def test_invalid(a):
    with pytest.raises(ValueError):
        lab.classify(a)


def test_replay_and_complete_denominators():
    a, b = lab.run_lab(repetitions=2), lab.run_lab(repetitions=2)
    a.pop("performance")
    b.pop("performance")
    assert a == b
    assert len(a["cells"]) == 15
    for cell in a["cells"]:
        assert sum(cell["counts"].values()) == 2
        assert len(cell["records"]) == 2


@pytest.mark.parametrize("bad", [None, [np.nan], [1j], [1, 2, 3], [[1]], ["bad"]])
def test_invalid_solver_witness_unresolved(monkeypatch, bad):
    from types import SimpleNamespace

    monkeypatch.setattr(lab, "linprog", lambda *a, **k: SimpleNamespace(success=True, x=bad))
    assert lab.classify([[1, 2]])["status"] == "unresolved"


def test_solver_exception_unresolved(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("solver failure")

    monkeypatch.setattr(lab, "linprog", fail)
    assert lab.classify([[1, 2]])["status"] == "unresolved"


def test_primal_failure_can_still_separate(monkeypatch):
    real = lab.linprog
    calls = 0

    def first_failed(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("primal failure")
        return real(*a, **k)

    monkeypatch.setattr(lab, "linprog", first_failed)
    assert lab.classify([[1, 2]])["status"] == "separated"


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        lab.run_lab(repetitions=1, seed=seed)


def test_boolean_dimensions_rejected():
    with pytest.raises(ValueError):
        lab.wendel(True, 1)
