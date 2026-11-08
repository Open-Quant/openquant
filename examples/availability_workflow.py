"""Preserve source clocks, intersect splitter training rows, then fit a transform.

Installed-package mode:
    python examples/availability_workflow.py

For reviewing this isolated standard-library module without initializing the
native package (not a test of full package import):
    python examples/availability_workflow.py --source-tree

The aligned metadata below is supplied separately from cleaned feature values.
Real callers must retain stable row identity across cleaner sorting/filtering;
clean_ohlcv discards extra columns. A deterministic reconstruction's computed_at
is intentionally not used as the availability of its historical source inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_selector(*, source_tree: bool):
    if not source_tree:
        from openquant.availability import select_available_training

        return select_available_training
    source = Path(__file__).resolve().parents[1] / "python/openquant/availability.py"
    spec = importlib.util.spec_from_file_location("availability_example_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the local availability helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_available_training


def run_example(selector):
    # Indices refer to the SAME aligned rows after cleaning. Row3 is held out by
    # an existing splitter; the helper must not put it back into training even
    # though its timestamps alone would pass. This is not a new splitter.
    # Row3 is historical: the calculation illustrates fold-local transformation,
    # not a simulated prospective prediction for that earlier date.
    decisions = [
        "2020-01-01T09:00:00Z",
        "2020-01-02T09:00:00Z",
        "2020-01-03T09:00:00Z",
        "2020-01-01T09:00:00Z",
        "2020-01-02T09:00:00Z",
    ]
    label_ends = [
        "2020-01-02T09:00:00Z",
        "2020-01-03T09:00:00Z",
        "2020-01-05T09:00:00Z",
        "2020-01-02T09:00:00Z",
        "2020-01-03T09:00:00Z",
    ]
    label_publications = [
        "2020-01-02T09:00:00Z",
        "2020-01-06T09:00:00Z",  # Resolved earlier, but not usable at fit.
        "2020-01-05T09:00:00Z",  # Equal boundary: conservative default excludes.
        "2020-01-02T09:00:00Z",
        "2020-01-03T09:00:00Z",
    ]
    source_ready = list(decisions)
    source_ready[4] = "2020-01-04T09:00:00Z"  # Retrospective revision, after row decision.
    fit = "2020-01-05T09:00:00Z"
    preexisting_train = [4, 2, 1, 0]
    raw_features = [10.0, 20.0, 30.0, 999.0, 50.0]
    selection = selector(
        decisions,
        label_ends,
        label_publications,
        source_ready,
        fit_time=fit,
        clock="utc",
        candidate_indices=preexisting_train,
    )
    assert selection.eligible_indices == (0,)
    assert 3 not in selection.eligible_indices
    # Fit a learned transform only AFTER filtering and intersecting the split.
    # Its fitted_at is fit, not each historical row's decision. That is valid
    # fold-local learning; do not substitute fitted_at for source_ready above.
    center = sum(raw_features[i] for i in selection.eligible_indices) / len(
        selection.eligible_indices
    )
    return {
        "selection": selection._asdict(),
        "transform_fitted_at": fit,
        "training_only_center": center,
        "heldout_index": 3,
        "centered_heldout_feature": raw_features[3] - center,
        "limits": "Supplied-clock audit and split intersection, not purging or a performance estimate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-tree",
        action="store_true",
        help="load only the local stdlib module; does not validate native package initialization",
    )
    args = parser.parse_args()
    result = run_example(load_selector(source_tree=args.source_tree))
    result["import_mode"] = "isolated_source_module" if args.source_tree else "installed_package"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
