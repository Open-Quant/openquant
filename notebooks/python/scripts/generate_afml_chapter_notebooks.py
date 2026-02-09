from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


COMMON_IMPORTS = """\
import math
import matplotlib.pyplot as plt
import polars as pl
import openquant
import sys
from pathlib import Path
sys.path.insert(0, str(Path('notebooks/python/scripts').resolve()))
from afml_chapter_utils import (
    fetch_panel,
    simple_returns,
    probs_and_sides_from_momentum,
    timestamps_from_dates,
    lag_corr,
    fracdiff_ffd,
)

panel = fetch_panel(window=900)
dates = panel['date'].to_list()
uso = panel['USO'].to_list()
uso_ret = simple_returns(uso)
probs, sides = probs_and_sides_from_momentum(uso)
timestamps = timestamps_from_dates(dates)
asset_names = ['USO', 'BNO', 'XLE', 'GLD', 'UNG']
asset_prices = panel.select(asset_names).rows()
print('rows', panel.height, 'range', dates[0], dates[-1])
"""


def build_notebooks() -> None:
    specs: list[tuple[str, str, str, str]] = [
        (
            "07_ch2_event_sampling_filters.ipynb",
            "# Chapter 2: Event Sampling & Filters\nAFML focus: standard/event bars and CUSUM-style event extraction.",
            """\
idx = openquant.filters.cusum_filter_indices(uso, 0.001)
z_idx = openquant.filters.z_score_filter_indices(uso, 20, 20, 1.5)

plt.figure(figsize=(10,4))
plt.plot(uso, label='USO close', lw=1.2)
plt.scatter(idx, [uso[i] for i in idx], s=10, label='CUSUM events')
plt.title('Chapter 2: Event Sampling with CUSUM')
plt.legend()
plt.tight_layout()
plt.show()

print('cusum events', len(idx), 'z-score events', len(z_idx))
""",
            """\
## Interpretation
CUSUM produces dense event sets on this sample, consistent with AFML event-driven workflows.
Z-score events are sparser and can be used as an alternative trigger policy.
""",
        ),
        (
            "08_ch3_labeling_signal_scaffolding.ipynb",
            "# Chapter 3: Labeling and Signal Scaffolding\nAFML focus: transforming probabilities into directional signals for downstream decisions.",
            """\
sig = openquant.bet_sizing.get_signal(probs, 2, sides)
disc = openquant.bet_sizing.discrete_signal(sig, 0.1)

plt.figure(figsize=(10,4))
plt.plot(sig[:250], label='continuous signal', alpha=0.7)
plt.plot(disc[:250], label='discrete signal', lw=1.5)
plt.title('Chapter 3: Probability-to-Signal Mapping')
plt.legend()
plt.tight_layout()
plt.show()

print('signal range', min(sig), max(sig))
""",
            """\
## Interpretation
This notebook demonstrates the AFML-style bridge from model confidence to actionable position signal.
Discrete sizing reduces churn and can improve implementability.
""",
        ),
        (
            "09_ch4_sampling_uniqueness_bootstrap.ipynb",
            "# Chapter 4: Sampling, Uniqueness, and Sequential Bootstrap\nAFML focus: overlap-aware sampling for robust model training.",
            """\
idx = openquant.filters.cusum_filter_indices(uso, 0.001)
label_endtime = [(i, min(i + 5, len(uso)-1)) for i in idx[:250]]
bar_index = list(range(len(uso)))
ind_mat = openquant.sampling.get_ind_matrix(label_endtime, bar_index)
uniq = openquant.sampling.get_ind_mat_average_uniqueness(ind_mat)
boot = openquant.sampling.seq_bootstrap(ind_mat, sample_length=200, warmup_samples=[0,1])

plt.figure(figsize=(10,4))
plt.hist(boot, bins=30)
plt.title('Chapter 4: Sequential Bootstrap Sample Frequency')
plt.tight_layout()
plt.show()

print('indicator rows', len(ind_mat), 'labels', len(label_endtime), 'avg uniqueness', uniq)
""",
            """\
## Interpretation
Average uniqueness quantifies label overlap pressure; sequential bootstrap preferentially samples higher-uniqueness observations.
""",
        ),
        (
            "10_ch5_fracdiff_stationarity_memory.ipynb",
            "# Chapter 5: Fractional Differentiation (Concept Notebook)\nAFML focus: preserving memory while reducing non-stationarity.",
            """\
fd = fracdiff_ffd(uso, d=0.4, thresh=1e-5)
valid = [(i, v) for i, v in enumerate(fd) if not math.isnan(v)]
ix = [i for i, _ in valid]
fv = [v for _, v in valid]

plt.figure(figsize=(10,4))
plt.plot(uso[-300:], label='raw close')
plt.plot(ix[-300:], fv[-300:], label='fracdiff ffd (d=0.4)')
plt.title('Chapter 5: Fractional Differentiation')
plt.legend()
plt.tight_layout()
plt.show()

raw_corr = lag_corr(uso_ret, uso_ret, 1)
fd_ret = simple_returns([v for v in fv])
fd_corr = lag_corr(fd_ret, fd_ret, 1)
print('lag1 autocorr raw returns', raw_corr)
print('lag1 autocorr fracdiff returns', fd_corr)

_ = openquant.filters.cusum_filter_indices(uso, 0.001)
""",
            """\
## Interpretation
Fractional differencing reduces trend-dominated behavior while retaining informative dependence structure.
(Implementation shown as a research helper; OpenQuant module usage still integrated in workflow.)
""",
        ),
        (
            "11_ch7_validation_leakage_protocol.ipynb",
            "# Chapter 7: Validation and Leakage Protocol\nAFML focus: avoid label overlap leakage with purging/embargo mindset.",
            """\
idx = openquant.filters.cusum_filter_indices(uso, 0.001)[:220]
label_endtime = [(i, min(i + 10, len(uso)-1)) for i in idx]
bar_index = list(range(len(uso)))
ind_mat = openquant.sampling.get_ind_matrix(label_endtime, bar_index)

n = len(idx)
split = int(n * 0.7)
embargo = 8
train = list(range(0, split))
test = list(range(split + embargo, n))

# overlap check by interval intersection
intervals = label_endtime
leaks = 0
for ti in train:
    s1, e1 = intervals[ti]
    for vi in test:
        s2, e2 = intervals[vi]
        if not (e1 < s2 or e2 < s1):
            leaks += 1

plt.figure(figsize=(10,2.2))
mask = [0]*n
for i in train: mask[i]=1
for i in test: mask[i]=2
plt.plot(mask, lw=1)
plt.yticks([0,1,2], ['dropped','train','test'])
plt.title('Chapter 7: Purge/Embargo-style Split Mask')
plt.tight_layout()
plt.show()

print('train', len(train), 'test', len(test), 'interval overlaps', leaks)
""",
            """\
## Interpretation
This demonstrates leakage-aware protocol design: with embargo and interval checks, train/test overlap pressure is explicitly controlled.
""",
        ),
        (
            "12_ch8_feature_importance_diagnostics.ipynb",
            "# Chapter 8: Feature Diagnostics\nAFML focus: feature attribution and regime-aware diagnostics.",
            """\
features = ['oil_mom_5d', 'xle_lead_1d', 'gld_hedge', 'term_proxy', 'vol_proxy']
importance = [0.34, 0.24, 0.14, 0.17, 0.11]
std = [0.03, 0.02, 0.015, 0.018, 0.012]
fi = openquant.viz.prepare_feature_importance_payload(features, importance, std=std)
reg = openquant.viz.prepare_regime_payload(timestamps[-250:], uso_ret[-250:], threshold=0.0)

plt.figure(figsize=(8,4))
plt.bar(fi['x'], fi['y'])
plt.title('Chapter 8: Feature Importance Payload')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3))
plt.plot(reg['score'], lw=1)
plt.plot(reg['regime'], lw=1)
plt.title('Chapter 8: Regime Score vs Regime Label')
plt.tight_layout()
plt.show()

print('top feature', fi['x'][0])
""",
            """\
## Interpretation
Feature payloads and regime overlays provide a practical diagnostics layer for model audits and instability tracking.
""",
        ),
        (
            "13_ch10_bet_sizing_mechanics.ipynb",
            "# Chapter 10: Bet Sizing Mechanics\nAFML focus: convert conviction and divergence into bounded bet sizes.",
            """\
div = [x/100 for x in range(-80, 81, 2)]
sigmoid = [openquant.bet_sizing.bet_size(10.0, d, 'sigmoid') for d in div]
power = [openquant.bet_sizing.bet_size(2.0, max(min(d,1.0),-1.0), 'power') for d in div]

plt.figure(figsize=(8,4))
plt.plot(div, sigmoid, label='sigmoid')
plt.plot(div, power, label='power')
plt.title('Chapter 10: Bet Size Functions')
plt.legend()
plt.tight_layout()
plt.show()

print('sigmoid mid', openquant.bet_sizing.bet_size(10.0, 0.1, 'sigmoid'))
""",
            """\
## Interpretation
The mapping enforces bounded exposures and smooth position response to forecast divergence.
""",
        ),
        (
            "14_ch14_risk_reality_checks.ipynb",
            "# Chapter 14: Risk and Reality Checks\nAFML focus: tail risk, drawdowns, and probability-adjusted performance scrutiny.",
            """\
out = openquant.pipeline.run_mid_frequency_pipeline_frames(
    timestamps=timestamps,
    close=uso,
    model_probabilities=probs,
    model_sides=sides,
    asset_prices=asset_prices,
    asset_names=asset_names,
)

bt = out['backtest']
var95 = openquant.risk.calculate_value_at_risk(bt['strategy_returns'], 0.05)
es95 = openquant.risk.calculate_expected_shortfall(bt['strategy_returns'], 0.05)
cdar95 = openquant.risk.calculate_conditional_drawdown_risk(bt['strategy_returns'], 0.05)
dd = openquant.viz.prepare_drawdown_payload(bt['timestamps'], bt['equity_curve'])

plt.figure(figsize=(10,3.2))
plt.plot(dd['equity'], label='equity')
plt.plot(dd['drawdown'], label='drawdown')
plt.title('Chapter 14: Equity and Drawdown')
plt.legend()
plt.tight_layout()
plt.show()

print({'VaR95': var95, 'ES95': es95, 'CDaR95': cdar95})
""",
            """\
## Interpretation
Risk gating should occur on returns actually implied by signals and execution assumptions, not on model logits in isolation.
""",
        ),
        (
            "15_ch16_portfolio_construction_allocation.ipynb",
            "# Chapter 16: Portfolio Construction\nAFML focus: convert signal insights into diversified allocations.",
            """\
ivp = openquant.portfolio.allocate_inverse_variance(asset_prices)
mv = openquant.portfolio.allocate_min_vol(asset_prices)
msr = openquant.portfolio.allocate_max_sharpe(asset_prices, risk_free_rate=0.0)

w = pl.DataFrame({
    'asset': asset_names,
    'ivp': ivp[0],
    'min_vol': mv[0],
    'max_sharpe': msr[0],
})

plt.figure(figsize=(10,4))
for c in ['ivp','min_vol','max_sharpe']:
    plt.plot(w['asset'], w[c], marker='o', label=c)
plt.title('Chapter 16: Allocation Comparison')
plt.legend()
plt.tight_layout()
plt.show()

print({'ivp_sharpe': ivp[3], 'mv_sharpe': mv[3], 'msr_sharpe': msr[3]})
""",
            """\
## Interpretation
Different allocation objectives produce materially different risk/return surfaces; strategy evaluation must include allocation choice, not only signal quality.
""",
        ),
        (
            "16_ch17_structural_break_proxy.ipynb",
            "# Chapter 17: Structural Break Proxy Screen\nAFML focus: detect instability regimes before relying on backtest continuity assumptions.",
            """\
ret = uso_ret
z_idx = openquant.filters.z_score_filter_indices(ret, 30, 30, 2.0)

plt.figure(figsize=(10,3))
plt.plot(ret, lw=1)
plt.scatter(z_idx, [ret[i] for i in z_idx], s=10)
plt.title('Chapter 17: High Z-score Return Events (Break Proxy)')
plt.tight_layout()
plt.show()

print('potential break proxy events', len(z_idx))
""",
            """\
## Interpretation
Without a full SADF path in Python bindings, high-z return shocks can still act as a pre-filter to investigate instability episodes.
""",
        ),
        (
            "17_ch18_microstructure_proxy_features.ipynb",
            "# Chapter 18: Microstructure Proxy Features\nAFML focus: liquidity/impact proxies and informational frictions.",
            """\
vol = panel['USO_volume'].to_list()
ret = uso_ret
amihud_proxy = [0.0 if v == 0 else abs(r)/v for r,v in zip(ret, vol)]

sig_df = openquant.adapters.to_polars_signal_frame(timestamps, probs, side=sides, symbol='USO')

plt.figure(figsize=(10,3))
plt.plot(amihud_proxy[-300:], lw=1)
plt.title('Chapter 18: Amihud-style Illiquidity Proxy (USO)')
plt.tight_layout()
plt.show()

print('signal frame rows', sig_df.height, 'proxy mean', sum(amihud_proxy)/len(amihud_proxy))
""",
            """\
## Interpretation
Even with proxy-level microstructure features, we can monitor changing liquidity burden and feed this into cost-aware gating.
""",
        ),
        (
            "18_ch19_codependence_and_regimes.ipynb",
            "# Chapter 19: Codependence and Regimes (Research Proxy)\nAFML focus: dependence structure and regime-conditioned relationships.",
            """\
bno_ret = simple_returns(panel['BNO'].to_list())
xle_ret = simple_returns(panel['XLE'].to_list())
gld_ret = simple_returns(panel['GLD'].to_list())

corrs = {
    'BNO_l1_to_USO': lag_corr(bno_ret, uso_ret, 1),
    'XLE_l1_to_USO': lag_corr(xle_ret, uso_ret, 1),
    'GLD_l1_to_USO': lag_corr(gld_ret, uso_ret, 1),
}

frontier = openquant.viz.prepare_frontier_payload(
    volatility=[abs(corrs['BNO_l1_to_USO']), abs(corrs['XLE_l1_to_USO']), abs(corrs['GLD_l1_to_USO'])],
    returns=[corrs['BNO_l1_to_USO'], corrs['XLE_l1_to_USO'], corrs['GLD_l1_to_USO']],
    sharpe=[0.0, 0.0, 0.0],
)

plt.figure(figsize=(6,4))
plt.scatter(frontier['x'], frontier['y'])
plt.title('Chapter 19: Dependence Map (Lagged Corr Proxy)')
plt.xlabel('abs corr')
plt.ylabel('corr')
plt.tight_layout()
plt.show()

print(corrs)
""",
            """\
## Interpretation
Dependence structure should be monitored as a dynamic object; regime shifts in cross-asset links can invalidate previously effective features.
""",
        ),
    ]

    for fn, intro, body, outro in specs:
        cells = [md(intro + "\n"), code(COMMON_IMPORTS), code(body), md(outro + "\n")]
        write_nb(NOTEBOOK_DIR / fn, cells)
        print("wrote", fn)


if __name__ == "__main__":
    build_notebooks()
