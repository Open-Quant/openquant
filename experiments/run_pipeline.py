from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import polars as pl

# ensure local python/ package is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

import openquant


def _read_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _config_digest(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run(config_path: Path, out_root: Path) -> Path:
    cfg = _read_config(config_path)
    meta = cfg.get("meta", {})
    data_cfg = cfg.get("data", {})
    pipe_cfg = cfg.get("pipeline", {})
    costs_cfg = cfg.get("costs", {})
    gates_cfg = cfg.get("gates", {})

    seed = int(meta.get("seed", 7))
    n_bars = int(meta.get("n_bars", 192))
    asset_names = list(data_cfg.get("asset_names", ["CL", "NG", "RB", "GC"]))

    dataset = openquant.research.make_synthetic_futures_dataset(
        n_bars=n_bars,
        seed=seed,
        asset_names=asset_names,
    )

    merged_cfg = {}
    merged_cfg.update(pipe_cfg)
    merged_cfg.update(costs_cfg)
    merged_cfg.update(gates_cfg)

    out = openquant.research.run_flywheel_iteration(dataset, config=merged_cfg)

    run_name = str(meta.get("name", config_path.stem))
    digest = _config_digest(cfg)
    run_dir = out_root / f"{run_name}-{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_df: pl.DataFrame = out["summary"]
    summary_df.write_parquet(run_dir / "metrics.parquet")
    out["frames"]["events"].write_parquet(run_dir / "events.parquet")
    out["frames"]["signals"].write_parquet(run_dir / "signals.parquet")
    out["frames"]["weights"].write_parquet(run_dir / "weights.parquet")
    out["frames"]["backtest"].write_parquet(run_dir / "backtest.parquet")

    manifest = {
        "run_name": run_name,
        "config_path": str(config_path),
        "config_digest": digest,
        "git_sha": _git_sha(REPO_ROOT),
        "seed": seed,
        "n_bars": n_bars,
        "asset_names": asset_names,
        "python": sys.version.split()[0],
        "openquant_package": "openquant (local editable)",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    promotion = out["promotion"]
    decision_lines = [
        f"# Promotion Decision: {run_name}",
        "",
        f"- promote_candidate: `{promotion['promote_candidate']}`",
        f"- passed_realized_sharpe: `{promotion['passed_realized_sharpe']}`",
        f"- passed_net_sharpe: `{promotion['passed_net_sharpe']}`",
        f"- passed_alignment_guard: `{promotion['passed_alignment_guard']}`",
        f"- passed_event_order_guard: `{promotion['passed_event_order_guard']}`",
        "",
        "## Cost Snapshot",
        f"- turnover: `{out['costs']['turnover']:.6f}`",
        f"- realized_vol: `{out['costs']['realized_vol']:.6f}`",
        f"- estimated_total_cost: `{out['costs']['estimated_total_cost']:.6f}`",
        f"- gross_total_return: `{out['costs']['gross_total_return']:.6f}`",
        f"- net_total_return: `{out['costs']['net_total_return']:.6f}`",
        f"- net_sharpe: `{out['costs']['net_sharpe']:.6f}`",
    ]
    (run_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible OpenQuant pipeline experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("--out", type=Path, default=Path("experiments/artifacts"), help="Artifacts root")
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
