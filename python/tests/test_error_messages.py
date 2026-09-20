"""Core errors reach Python as ``ValueError`` carrying the core's own message (#35).

Before the typed errors, the bindings formatted errors with ``{:?}``: string errors arrived
wrapped in quotes and enum errors as a bare variant name such as ``InvalidRepeat``.
One call per bound module whose errors were retyped.
"""

import pytest

from openquant._core import (
    codependence,
    data,
    ensemble,
    microstructural,
    onc,
    risk,
    strategy_risk,
    synthetic_bt,
)

CASES = {
    "ensemble_empty": (
        lambda: ensemble.aggregate_regression_mean([]),
        "per_model_predictions cannot be empty",
    ),
    "ensemble_invalid": (
        lambda: ensemble.bagging_ensemble_variance(-1.0, 0.5, 10),
        "single_estimator_variance must be non-negative",
    ),
    "synthetic_bt": (
        lambda: synthetic_bt.calibrate_ou_params([1.0, 1.0]),
        "prices must include at least 3 observations",
    ),
    "data": (
        lambda: data.align_calendar([1], ["A"], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], 0),
        "interval_seconds must be > 0",
    ),
    "microstructural": (
        lambda: microstructural.encode_tick_rule_array([1, 7]),
        "Unknown value for tick rule: 7",
    ),
    "risk": (
        lambda: risk.calculate_value_at_risk([0.1, 0.2], 1.5),
        "confidence level must be in [0, 1]",
    ),
    "strategy_risk": (
        lambda: strategy_risk.sharpe_symmetric(1.5, 10.0),
        "invalid input: precision must be finite and in [0, 1]",
    ),
    "codependence": (
        lambda: codependence.distance_correlation([1.0, 2.0, 3.0], [1.0]),
        "the two series have different lengths",
    ),
    "onc": (
        lambda: onc.get_onc_clusters([[1.0, 0.2], [0.2, 1.0]], 0),
        "repeat must be positive",
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_core_message_reaches_python_verbatim(name):
    call, message = CASES[name]
    with pytest.raises(ValueError) as excinfo:
        call()
    assert str(excinfo.value) == message
