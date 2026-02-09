from __future__ import annotations

import subprocess
import sys

NOTEBOOKS = [
    "notebooks/python/06_afml_real_data_end_to_end.ipynb",
    "notebooks/python/07_ch2_event_sampling_filters.ipynb",
    "notebooks/python/08_ch3_labeling_signal_scaffolding.ipynb",
    "notebooks/python/09_ch4_sampling_uniqueness_bootstrap.ipynb",
    "notebooks/python/10_ch5_fracdiff_stationarity_memory.ipynb",
    "notebooks/python/11_ch7_validation_leakage_protocol.ipynb",
    "notebooks/python/12_ch8_feature_importance_diagnostics.ipynb",
    "notebooks/python/13_ch10_bet_sizing_mechanics.ipynb",
    "notebooks/python/14_ch14_risk_reality_checks.ipynb",
    "notebooks/python/15_ch16_portfolio_construction_allocation.ipynb",
    "notebooks/python/16_ch17_structural_break_proxy.ipynb",
    "notebooks/python/17_ch18_microstructure_proxy_features.ipynb",
    "notebooks/python/18_ch19_codependence_and_regimes.ipynb",
]


def main() -> None:
    for nb in NOTEBOOKS:
        print(f"EXEC {nb}")
        cmd = [sys.executable, "notebooks/python/scripts/execute_notebook_cells.py", nb]
        subprocess.run(cmd, check=True)
    print("ALL_NOTEBOOKS_OK")


if __name__ == "__main__":
    main()
