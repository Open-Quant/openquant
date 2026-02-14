from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import polars as pl
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

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


def _drawdown_from_equity(equity: Sequence[float]) -> list[float]:
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
        y_vals = [
            pad + ((max_v - value) / (max_v - min_v)) * plot_h
            for value in values
        ]

    commands = [f"M {x_vals[0]:.2f} {y_vals[0]:.2f}"]
    commands.extend(
        f"L {x_vals[idx]:.2f} {y_vals[idx]:.2f}"
        for idx in range(1, len(x_vals))
    )
    return " ".join(commands)


def _write_line_chart_svg(
    *,
    values: Sequence[float],
    output_path: Path,
    title: str,
    stroke: str,
) -> None:
    width = 960
    height = 420
    pad = 44
    path_d = _line_path(values, width=width, height=height, pad=pad)
    y_axis = f"M {pad:.2f} {pad:.2f} L {pad:.2f} {height - pad:.2f}"
    x_axis = f"M {pad:.2f} {height - pad:.2f} L {width - pad:.2f} {height - pad:.2f}"
    min_v = min(values) if values else 0.0
    max_v = max(values) if values else 0.0
    body = "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-labelledby='chart-title'>",
            "<rect width='100%' height='100%' fill='#ffffff'/>",
            f"<text id='chart-title' x='{pad}' y='24' font-family='monospace' font-size='16' fill='#111827'>{title}</text>",
            f"<text x='{pad}' y='44' font-family='monospace' font-size='12' fill='#374151'>min={min_v:.6f} max={max_v:.6f}</text>",
            f"<path d='{y_axis}' fill='none' stroke='#9ca3af' stroke-width='1'/>",
            f"<path d='{x_axis}' fill='none' stroke='#9ca3af' stroke-width='1'/>",
            f"<path d='{path_d}' fill='none' stroke='{stroke}' stroke-width='2'/>",
            "</svg>",
            "",
        ]
    )
    output_path.write_text(body, encoding="utf-8")


def _write_plot_artifacts(backtest: pl.DataFrame, run_dir: Path) -> None:
    equity = [float(x) for x in backtest["equity"].to_list()]
    drawdown = _drawdown_from_equity(equity)
    _write_line_chart_svg(
        values=equity,
        output_path=run_dir / "equity_curve.svg",
        title="Equity Curve",
        stroke="#1d4ed8",
    )
    _write_line_chart_svg(
        values=drawdown,
        output_path=run_dir / "drawdown.svg",
        title="Drawdown",
        stroke="#b91c1c",
    )


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
    _write_plot_artifacts(out["frames"]["backtest"], run_dir)

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
