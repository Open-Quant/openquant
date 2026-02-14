from __future__ import annotations

import math

import polars as pl

import openquant


def _make_base_frame(n: int = 720, seed: int = 11) -> pl.DataFrame:
    ds = openquant.research.make_synthetic_futures_dataset(n_bars=n, seed=seed, asset_names=["SPY", "QQQ", "IWM"])
    volume = [float(80_000 + int(25_000 * (1.0 + math.sin(i / 17.0)))) for i in range(n)]
    outlier_idx = max(n // 3, 1)
    volume[outlier_idx] *= 25.0
    return pl.DataFrame(
        {
            "ts": ds.timestamps,
            "symbol": ["SPY"] * n,
            "open": ds.close,
            "high": [c * 1.002 for c in ds.close],
            "low": [c * 0.998 for c in ds.close],
            "close": ds.close,
            "volume": volume,
            "adj_close": ds.close,
        }
    )


def main() -> None:
    base = _make_base_frame()
    variants = {
        "time_1h": openquant.bars.build_time_bars(base, interval="1h"),
        "tick_40": openquant.bars.build_tick_bars(base, ticks_per_bar=40),
        "volume_2m": openquant.bars.build_volume_bars(base, volume_per_bar=2_000_000),
        "dollar_200m": openquant.bars.build_dollar_bars(base, dollar_value_per_bar=200_000_000),
    }

    print("bar_family,n_bars,lag1_return_autocorr,lag1_sq_return_autocorr,return_std")
    for name, frame in variants.items():
        d = openquant.bars.bar_diagnostics(frame)
        print(
            f"{name},{int(d['n_bars'])},"
            f"{d['lag1_return_autocorr']:.6f},"
            f"{d['lag1_sq_return_autocorr']:.6f},"
            f"{d['return_std']:.6f}"
        )


if __name__ == "__main__":
    main()
