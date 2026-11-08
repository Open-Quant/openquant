"""Execute the notebooks' actual registry key expressions without their Monte Carlo runs."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import openquant
import polars as pl
import pytest
from openquant import evaluation

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks/python"


def _source(number):
    path = next(NOTEBOOKS.glob(f"{number}_*.ipynb"))
    cells = json.loads(path.read_text())["cells"]
    return ast.parse("\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"))


def _namespace(number):
    values = {
        "VOL_SPAN": 50,
        "CUSUM_MULT": 2.0,
        "PT_SL": 1.5,
        "HORIZON_DAYS": 14,
        "FAST": 10,
        "SLOW": 40,
        "L2": 1.0,
        "COST_BPS": 5.0,
        "N_SPLITS": 5,
        "PCT_EMBARGO": 0.01,
        "FEATURES": ("vol_ratio", "efficiency", "ma_gap"),
        "FEATURE_SETS": {"all": [0, 1, 2]},
        "SEED": 47,
        "BAGGING": {"n_estimators": 20, "max_samples": 0.2},
        "WARM": 120,
        "MAX_LOOKBACK": 120,
        "N_GROUPS": 10,
        "K_TEST": 3,
        "WF_TRAIN": 504,
        "WF_STEP": 252,
        "HORIZON": 5,
        "WARMUP": 100,
        "FFD_THRESH": 0.001,
        "D_GRID": [0.0, 0.5, 1.0],
        "N_LAGS": 5,
        "DIFF_HORIZONS": (1, 5, 20, 60),
        "N_REPS": 30,
        "ANN": 252,
        "STRENGTHS": [0.0],
        "name": "null" if number == 12 else "ffd_dstar",
        "k": 0,
        "raw_returns": np.array([0.1, -0.2, 0.3, 0.4]),
    }
    series = {"A": (["2024-01-01", "2024-01-02"], np.array([100.0, 101.0]))}
    values["protocol"] = None
    values.update(
        openquant=openquant,
        evaluation=evaluation,
        ev=evaluation,
        np=np,
        pl=pl,
        series=series,
        planted=series,
        control=series,
        data_label="same-label",
        PLANTED_LABEL="same-label",
        CONTROL_LABEL="same-label",
        ev_list=[{"logp": np.array([1.0, 1.1, 1.3, 1.2])}],
        cfg={"model": "logit", "cv": "purged-kfold", "features": "all", "fast": 2},
    )
    tree = _source(number)
    helpers = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "trial_identity"
    ]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "<trial identity>", "exec"), values)
    return values


def _key(number, ns, receiver=None):
    tree = _source(number)
    if number == 12:
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "registry_for"
            and isinstance(n.args[0], ast.Name)
        )
        ns["dataset"] = eval(
            compile(ast.Expression(call.args[2]), "<dataset identity>", "eval"), ns
        )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record"
        and (receiver is None or isinstance(n.func.value, ast.Name) and n.func.value.id == receiver)
    ]
    return eval(compile(ast.Expression(calls[0].args[0]), "<registry key>", "eval"), ns)


@pytest.mark.parametrize("number", [11, 12, 13, 14])
def test_same_label_different_actual_data_is_a_distinct_trial(number, tmp_path):
    ns = _namespace(number)
    registry = evaluation.TrialRegistry(tmp_path / "trials.json")
    first = _key(number, ns)
    registry.record(first, [0.1, -0.2, 0.3, 0.4])
    registry.record(_key(number, ns), [0.1, -0.2, 0.3, 0.4])
    assert registry.n_trials == 1
    ns["series"]["A"][1][1] = 102.0
    ns["raw_returns"][1] = -0.25
    ns["ev_list"][0]["logp"][1] = 1.15
    registry.record(_key(number, ns), [0.1, -0.25, 0.3, 0.4])
    assert registry.n_trials == 2


@pytest.mark.parametrize(
    "number,parameter", [(11, "COST_BPS"), (12, "PCT_EMBARGO"), (13, "L2"), (14, "HORIZON")]
)
def test_protocol_change_preserves_prior_trial(number, parameter, tmp_path):
    ns = _namespace(number)
    registry = evaluation.TrialRegistry(tmp_path / "trials.json")
    registry.record(_key(number, ns), [0.1, -0.2, 0.3, 0.4])
    ns[parameter] *= 2
    registry.record(_key(number, ns), [0.1, -0.2, 0.3, 0.4])
    assert registry.n_trials == 2


@pytest.mark.parametrize("number", [11, 12, 13, 14])
def test_display_and_replication_counts_do_not_change_trial_identity(number):
    ns = _namespace(number)
    before = copy.deepcopy(_key(number, ns))
    ns["N_REPS"] = 999
    ns["ANN"] = 999
    assert _key(number, ns) == before


@pytest.mark.parametrize(
    "number,parameter",
    [
        (n, p)
        for n in (11, 13)
        for p in (
            "VOL_SPAN",
            "PT_SL",
            "HORIZON_DAYS",
            "FAST",
            "SLOW",
            "COST_BPS",
            "N_SPLITS",
            "PCT_EMBARGO",
            "L2",
        )
    ]
    + [
        (12, p)
        for p in (
            "WARM",
            "MAX_LOOKBACK",
            "N_GROUPS",
            "K_TEST",
            "PCT_EMBARGO",
            "WF_TRAIN",
            "WF_STEP",
        )
    ]
    + [
        (14, p)
        for p in (
            "HORIZON",
            "WARMUP",
            "L2",
            "COST_BPS",
            "N_SPLITS",
            "PCT_EMBARGO",
            "N_LAGS",
            "FFD_THRESH",
        )
    ],
)
def test_effective_protocol_fields_change_the_key(number, parameter):
    ns = _namespace(number)
    before = evaluation.config_hash(_key(number, ns))
    ns[parameter] *= 2
    assert evaluation.config_hash(_key(number, ns)) != before


@pytest.mark.parametrize("number", [11, 13])
def test_exact_price_bits_and_timestamps_are_part_of_identity(number):
    ns = _namespace(number)
    before = evaluation.config_hash(_key(number, ns))
    ns["series"]["A"][1][1] = np.nextafter(101.0, np.inf)
    assert evaluation.config_hash(_key(number, ns)) != before
    ns["series"]["A"][1][1] = 101.0
    ns["series"]["A"][0][1] = "2024-01-03"
    assert evaluation.config_hash(_key(number, ns)) != before


def test_bagging_seed_and_parameters_are_captured_but_not_for_logit():
    ns = _namespace(11)
    before = evaluation.config_hash(_key(11, ns))
    ns["SEED"] += 1
    ns["BAGGING"]["n_estimators"] += 1
    assert evaluation.config_hash(_key(11, ns)) == before
    ns["cfg"]["model"] = "sb-bagging"
    before = evaluation.config_hash(_key(11, ns))
    ns["SEED"] += 1
    assert evaluation.config_hash(_key(11, ns)) != before
    before = evaluation.config_hash(_key(11, ns))
    ns["BAGGING"]["n_estimators"] += 1
    assert evaluation.config_hash(_key(11, ns)) != before


@pytest.mark.parametrize("number", [11, 13])
def test_primary_trial_ignores_unused_model_and_cv_settings(number):
    ns = _namespace(number)
    ns["cfg"]["cv"] = None
    before = evaluation.config_hash(_key(number, ns))
    ns["L2"] *= 2
    ns["N_SPLITS"] += 1
    assert evaluation.config_hash(_key(number, ns)) == before


def test_fixed_ffd_does_not_inherit_the_dstar_search_grid():
    ns = _namespace(14)
    ns["name"] = "ffd_d0.4"
    before = evaluation.config_hash(_key(14, ns))
    ns["D_GRID"].append(0.7)
    assert evaluation.config_hash(_key(14, ns)) == before
    ns["name"] = "ffd_dstar"
    before = evaluation.config_hash(_key(14, ns))
    ns["D_GRID"].append(0.8)
    assert evaluation.config_hash(_key(14, ns)) != before


def test_versioned_identity_keeps_legacy_records_and_top_level_model(tmp_path):
    ns = _namespace(11)
    registry = evaluation.TrialRegistry(tmp_path / "trials.json")
    legacy = {"data": ns["data_label"], **ns["cfg"]}
    registry.record(legacy, [0.1, -0.2, 0.3, 0.4])
    key = _key(11, ns)
    assert key["model"] == "logit"
    assert key["trial_identity"]["version"] == 1
    registry.record(key, [0.1, -0.2, 0.3, 0.4])
    assert registry.n_trials == 2
    assert registry.trials[0].config == legacy
    registry.record(key, [0.1, -0.2, 0.3, 0.4])
    assert registry.n_trials == 2


def test_notebook_namespace_separates_otherwise_matching_protocols():
    first = _namespace(11)
    second = _namespace(13)
    assert (
        _key(11, first)["trial_identity"]["notebook"]
        != _key(13, second)["trial_identity"]["notebook"]
    )


def test_planted_registration_allows_existing_history(tmp_path):
    ns = _namespace(13)
    registry = evaluation.TrialRegistry(tmp_path / "planted.json")
    registry.record({"historical": True}, [0.1, -0.2, 0.3, 0.4])
    ns.update(
        registry_planted=registry,
        GRID=[ns["cfg"]],
        rows_h=[{}],
        kept_h={"one": [None, None, np.array([0.1, -0.2, 0.3, 0.4])]},
        name=lambda cfg: "one",
        net=lambda returns: returns,
        no_bet_planted=[],
        current_planted_keys=set(),
    )
    tree = _source(13)
    loop = next(
        n
        for n in tree.body
        if isinstance(n, ast.For)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "registry_planted"
            and c.func.attr == "record"
            for c in ast.walk(n)
        )
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "<registration loop>", "exec"), ns)
    ns["N_TRIALS_PLANTED"] = registry.n_trials
    assertions = [
        n
        for n in tree.body
        if isinstance(n, ast.Assert)
        and any(
            isinstance(c, ast.Name) and c.id in {"N_TRIALS_PLANTED", "current_planted_keys"}
            for c in ast.walk(n)
        )
    ]
    assert assertions
    exec(compile(ast.Module(body=assertions, type_ignores=[]), "<history assertions>", "exec"), ns)
    assert registry.n_trials == 2
    assert registry.trials[0].config == {"historical": True}


def test_data_section_uses_its_own_validation_protocol(tmp_path):
    ns = _namespace(12)
    ns.update(
        Path=Path,
        REGISTRY_DIR=tmp_path,
        CONFIGS=[ns["cfg"]],
        R_data=np.array([[0.1], [-0.2], [0.3], [0.4]]),
        T_DATA=4,
        DATA_GROUPS=6,
        DATA_K=2,
        meta={"source": "test", "dataset_hash": "sha256:test"},
    )
    tree = _source(12)
    definition = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "registry_for"
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<registry_for>", "exec"), ns)
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "registry_for"
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "data"
    )
    expression = compile(ast.Expression(call), "<data registry call>", "eval")
    first = eval(expression, ns)
    protocol = first.trials[0].config["trial_identity"]["protocol"]
    assert (
        protocol["n_groups"],
        protocol["k_test"],
        protocol["wf_train"],
        protocol["wf_step"],
    ) == (6, 2, 2, 1)
    ns["DATA_GROUPS"] = 7
    assert eval(expression, ns).n_trials == 2
    ns["N_GROUPS"] = 50
    ns["WF_TRAIN"] = 1000
    assert eval(expression, ns).n_trials == 2


def test_bet_sizing_trial_captures_event_sampling_threshold():
    ns = _namespace(13)
    before = evaluation.config_hash(_key(13, ns))
    ns["CUSUM_MULT"] *= 2
    assert evaluation.config_hash(_key(13, ns)) != before


@pytest.mark.parametrize("number", [11, 12, 13, 14])
def test_sequence_permutations_do_not_collide_in_order_invariant_frame_hash(number):
    ns = _namespace(number)
    before = evaluation.config_hash(_key(number, ns))
    timestamps, prices = ns["series"]["A"]
    ns["series"]["A"] = (timestamps[::-1], prices[::-1])
    ns["raw_returns"] = ns["raw_returns"][::-1]
    ns["ev_list"][0]["logp"] = ns["ev_list"][0]["logp"][::-1]
    assert evaluation.config_hash(_key(number, ns)) != before
