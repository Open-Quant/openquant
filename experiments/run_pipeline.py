from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Sequence

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


def _dataset_frame(dataset) -> pl.DataFrame:
    """The research dataset as one table, so its content hash covers every input."""
    columns = {
        "ts": dataset.timestamps,
        "close": dataset.close,
        "model_probability": dataset.model_probabilities,
        "model_side": dataset.model_sides,
    }
    for j, name in enumerate(dataset.asset_names):
        columns[f"asset:{name}"] = [row[j] for row in dataset.asset_prices]
    return pl.DataFrame(columns)


def _dataset_from_cfg(cfg: dict) -> tuple[object, dict]:
    meta = cfg.get("meta", {})
    data_cfg = cfg.get("data", {})
    seed = int(meta.get("seed", 7))
    n_bars = int(meta.get("n_bars", 192))
    asset_names = list(data_cfg.get("asset_names", ["CL", "NG", "RB", "GC"]))
    dataset = openquant.research.make_synthetic_futures_dataset(
        n_bars=n_bars,
        seed=seed,
        asset_names=asset_names,
    )
    dataset_meta = {
        "source": "openquant.research.make_synthetic_futures_dataset",
        "seed": seed,
        "n_bars": n_bars,
        "asset_names": asset_names,
        # One-minute bars: Sharpe ratios and volatility are annualised with 390 * 252 bars.
        "periods_per_year": dataset.periods_per_year,
    }
    # Every manifest records the content hash of the exact data the run used.
    dataset_meta["hash"] = openquant.data.dataset_hash(_dataset_frame(dataset))
    return dataset, dataset_meta


def _merged_run_cfg(cfg: dict) -> dict:
    merged_cfg: dict = {}
    merged_cfg.update(cfg.get("pipeline", {}))
    merged_cfg.update(cfg.get("costs", {}))
    merged_cfg.update(cfg.get("gates", {}))
    return merged_cfg


def _drawdown_from_equity(equity: Sequence[float]) -> list[float]:
    """Fractional drawdown from the running peak: equity / peak - 1 (0 while peak <= 0)."""
    peak = float("-inf")
    drawdown: list[float] = []
    for value in equity:
        peak = max(peak, value)
        drawdown.append((value / peak) - 1.0 if peak > 0.0 else 0.0)
    return drawdown


def _line_path(values: Sequence[float], width: int, height: int, pad: int) -> str:
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    if not values:
        mid_y = pad + (plot_h / 2.0)
        return f"M {pad:.2f} {mid_y:.2f} L {width - pad:.2f} {mid_y:.2f}"

    min_v = min(values)
    max_v = max(values)
    if len(values) == 1:
        x_vals = [pad + (plot_w / 2.0)]
    else:
        step = plot_w / float(len(values) - 1)
        x_vals = [pad + (i * step) for i in range(len(values))]

    if max_v == min_v:
        y_vals = [pad + (plot_h / 2.0) for _ in values]
    else:
        y_vals = [pad + ((max_v - value) / (max_v - min_v)) * plot_h for value in values]

    commands = [f"M {x_vals[0]:.2f} {y_vals[0]:.2f}"]
    commands.extend(f"L {x_vals[idx]:.2f} {y_vals[idx]:.2f}" for idx in range(1, len(x_vals)))
    return " ".join(commands)


def _write_line_chart_svg(
    *,
    values: Sequence[float],
    output_path: Path,
    title: str,
    stroke: str,
) -> None:
    """Write a fixed-size SVG line chart. Same values in, same bytes out."""
    width = 960
    height = 420
    pad = 44
    path_d = _line_path(values, width=width, height=height, pad=pad)
    y_axis = f"M {pad:.2f} {pad:.2f} L {pad:.2f} {height - pad:.2f}"
    x_axis = f"M {pad:.2f} {height - pad:.2f} L {width - pad:.2f} {height - pad:.2f}"
    min_v = min(values) if values else 0.0
    max_v = max(values) if values else 0.0
    title_text = html.escape(title)
    body = "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}' role='img' aria-labelledby='chart-title'>",
            "<rect width='100%' height='100%' fill='#ffffff'/>",
            f"<text id='chart-title' x='{pad}' y='24' font-family='monospace' font-size='16' "
            f"fill='#111827'>{title_text}</text>",
            f"<text x='{pad}' y='40' font-family='monospace' font-size='12' fill='#374151'>"
            f"min={min_v:.6f} max={max_v:.6f} n={len(values)}</text>",
            f"<path d='{y_axis}' fill='none' stroke='#9ca3af' stroke-width='1'/>",
            f"<path d='{x_axis}' fill='none' stroke='#9ca3af' stroke-width='1'/>",
            f"<path d='{path_d}' fill='none' stroke='{stroke}' stroke-width='2'/>",
            "</svg>",
            "",
        ]
    )
    output_path.write_text(body, encoding="utf-8")


def _write_plot_artifacts(backtest: pl.DataFrame, run_dir: Path, run_name: str) -> None:
    """Write equity_curve.svg and drawdown.svg from the backtest frame's `equity` column."""
    equity = [float(x) for x in backtest["equity"].to_list()]
    _write_line_chart_svg(
        values=equity,
        output_path=run_dir / "equity_curve.svg",
        title=f"Equity Curve: {run_name}",
        stroke="#1d4ed8",
    )
    _write_line_chart_svg(
        values=_drawdown_from_equity(equity),
        output_path=run_dir / "drawdown.svg",
        title=f"Drawdown: {run_name}",
        stroke="#b91c1c",
    )


def _write_run_artifacts(
    run_dir: Path,
    *,
    run_name: str,
    config_path: Path,
    manifest_cfg: dict,
    dataset_meta: dict,
    out: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_df: pl.DataFrame = out["summary"]
    summary_df.write_parquet(run_dir / "metrics.parquet")
    out["frames"]["events"].write_parquet(run_dir / "events.parquet")
    out["frames"]["signals"].write_parquet(run_dir / "signals.parquet")
    out["frames"]["weights"].write_parquet(run_dir / "weights.parquet")
    out["frames"]["backtest"].write_parquet(run_dir / "backtest.parquet")
    _write_plot_artifacts(out["frames"]["backtest"], run_dir, run_name)

    run_manifest = openquant.research.research_run_manifest(
        manifest_cfg,
        dataset_meta={**dataset_meta, "config_path": str(config_path)},
    )
    manifest = {
        "run_name": run_name,
        "config_path": str(config_path),
        "config_digest": run_manifest["config_digest"],
        "git_sha": _git_sha(REPO_ROOT),
        "python": sys.version.split()[0],
        "openquant_package": "openquant (local editable)",
        "config": manifest_cfg,
    }
    openquant.data.record_dataset_hash(manifest, digest=dataset_meta["hash"], **dataset_meta)
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


def run(config_path: Path, out_root: Path) -> Path:
    cfg = _read_config(config_path)
    meta = cfg.get("meta", {})
    dataset, dataset_meta = _dataset_from_cfg(cfg)
    merged_cfg = _merged_run_cfg(cfg)

    out = openquant.research.run_flywheel_iteration(dataset, config=merged_cfg)

    run_name = str(meta.get("name", config_path.stem))
    digest = openquant.research.research_run_manifest(cfg)["config_digest"]
    run_dir = out_root / f"{run_name}-{digest}"
    _write_run_artifacts(
        run_dir,
        run_name=run_name,
        config_path=config_path,
        manifest_cfg=cfg,
        dataset_meta=dataset_meta,
        out=out,
    )

    print(run_dir)
    return run_dir


def run_grid(config_path: Path, grid_config_path: Path, out_root: Path) -> Path:
    base_cfg = _read_config(config_path)
    grid_cfg = _read_config(grid_config_path)
    runs = list(grid_cfg.get("runs", []))
    if not runs:
        raise ValueError("grid config must define at least one [[runs]] entry")

    dataset, dataset_meta = _dataset_from_cfg(base_cfg)
    base_run_cfg = _merged_run_cfg(base_cfg)
    run_cfgs: list[dict] = []
    run_names: list[str] = []

    for idx, run_item in enumerate(runs):
        item = dict(run_item)
        run_name = str(item.pop("name", f"run_{idx:02d}"))
        cfg = dict(base_run_cfg)
        cfg.update(item)
        run_cfgs.append(cfg)
        run_names.append(run_name)

    grid_out = openquant.research.run_flywheel_grid(dataset, run_cfgs, run_names=run_names)
    digest = openquant.research.research_run_manifest(
        {"base_config": base_cfg, "grid_config": grid_cfg}
    )["config_digest"]
    run_set_name = str(base_cfg.get("meta", {}).get("name", config_path.stem))
    run_dir = out_root / f"{run_set_name}-grid-{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: pl.DataFrame = grid_out["leaderboard"]
    leaderboard.write_parquet(run_dir / "leaderboard.parquet")

    for run in grid_out["runs"]:
        per_run_name = str(run["run_name"])
        per_run_cfg = dict(run["config"])
        per_run_out = run["output"]
        per_digest = openquant.research.research_run_manifest(per_run_cfg)["config_digest"]
        per_dir = run_dir / f"{per_run_name}-{per_digest}"
        _write_run_artifacts(
            per_dir,
            run_name=per_run_name,
            config_path=config_path,
            manifest_cfg={"base_config_path": str(config_path), "grid_entry": per_run_cfg},
            dataset_meta=dataset_meta,
            out=per_run_out,
        )

    run_manifest = {
        "mode": "grid",
        "config_path": str(config_path),
        "grid_config_path": str(grid_config_path),
        "config_digest": digest,
        "git_sha": _git_sha(REPO_ROOT),
        "python": sys.version.split()[0],
        "run_count": len(run_cfgs),
        "run_names": run_names,
    }
    openquant.data.record_dataset_hash(run_manifest, digest=dataset_meta["hash"], **dataset_meta)
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible OpenQuant pipeline experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help="Optional TOML file with [[runs]] parameter overrides for multi-run sweeps",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/artifacts"), help="Artifacts root")
    args = parser.parse_args()
    if args.grid_config is not None:
        run_grid(args.config, args.grid_config, args.out)
    else:
        run(args.config, args.out)


if __name__ == "__main__":
    main()
