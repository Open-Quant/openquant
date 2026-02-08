from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

import openquant


def main() -> None:
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=160, seed=11)
    out = openquant.research.run_flywheel_iteration(ds)

    assert out["frames"]["events"].height > 5
    assert out["frames"]["signals"].height == len(ds.timestamps)
    assert out["frames"]["weights"]["weight"].sum() == out["frames"]["weights"]["weight"].sum()

    label_endtime = [(0, 3), (1, 4), (2, 6), (5, 8)]
    bar_index = list(range(12))
    ind_mat = openquant.sampling.get_ind_matrix(label_endtime, bar_index)
    samples = openquant.sampling.seq_bootstrap(ind_mat, sample_length=6, warmup_samples=[0, 1])
    assert len(samples) == 6

    payload = openquant.viz.prepare_feature_importance_payload(
        ["term_structure", "roll_yield", "carry", "momentum"],
        [0.32, 0.28, 0.19, 0.21],
    )
    assert payload["chart"] == "bar"

    print("python notebook smoke: ok")


if __name__ == "__main__":
    main()
