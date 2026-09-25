import sys as _sys

from . import _core, adapters, bars, data, evaluation, feature_diagnostics, pipeline, research, viz

# Compiled submodules re-exported as-is. `bars`, `data` and `pipeline` are
# deliberately absent: the pure-Python modules imported above wrap the compiled
# ones (still reachable as `openquant._core.<name>`).
risk = _core.risk
filters = _core.filters
sampling = _core.sampling
labeling = _core.labeling
bet_sizing = _core.bet_sizing
portfolio = _core.portfolio
fracdiff = _core.fracdiff
fast_ewma = _core.fast_ewma
volatility = _core.volatility
codependence = _core.codependence
backtest_stats = _core.backtest_stats
sample_weights = _core.sample_weights
microstructural = _core.microstructural
strategy_risk = _core.strategy_risk
ensemble = _core.ensemble
structural_breaks = _core.structural_breaks
synthetic_bt = _core.synthetic_bt
ef3m = _core.ef3m
streaming_hpc = _core.streaming_hpc
hrp = _core.hrp
hcaa = _core.hcaa
onc = _core.onc
cla = _core.cla
sb_bagging = _core.sb_bagging
dynamic_allocation = _core.dynamic_allocation

_CORE_REEXPORTS = [
    "risk",
    "filters",
    "sampling",
    "labeling",
    "bet_sizing",
    "portfolio",
    "fracdiff",
    "fast_ewma",
    "volatility",
    "codependence",
    "backtest_stats",
    "sample_weights",
    "microstructural",
    "strategy_risk",
    "ensemble",
    "structural_breaks",
    "synthetic_bt",
    "ef3m",
    "streaming_hpc",
    "hrp",
    "hcaa",
    "onc",
    "cla",
    "sb_bagging",
    "dynamic_allocation",
]

# Compiled submodules are attributes, not files, so `import openquant.hrp`
# only works once they are registered with the import system.
for _name in _CORE_REEXPORTS:
    _sys.modules.setdefault(f"{__name__}.{_name}", getattr(_core, _name))
del _name

__all__ = [
    *_CORE_REEXPORTS,
    "bars",
    "data",
    "evaluation",
    "feature_diagnostics",
    "pipeline",
    "research",
    "adapters",
    "viz",
]
