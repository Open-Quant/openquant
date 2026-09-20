"""Reference values for get_daily_vol: AFML snippet 3.1, run in pandas.

    uv run --with pandas python tests/fixtures/volatility/generate.py

Independent of this library. Writes daily_vol_reference.json next to this file.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
close = pd.read_csv(HERE.parent / "filters" / "dollar_bar_sample.csv", index_col=0, parse_dates=[0])["close"]


def get_daily_vol(close, lookback=100):
    # Snippet 3.1, verbatim apart from names.
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=lookback).std()


vol = get_daily_vol(close, 100)
fmt = "%Y-%m-%d %H:%M:%S.%f"
picks = [0, 1, 2, 10, 100, len(vol) // 2, len(vol) - 1]
out = {
    "source": "AFML snippet 3.1 in pandas %s on tests/fixtures/filters/dollar_bar_sample.csv" % pd.__version__,
    "lookback": 100,
    "n_bars": int(close.shape[0]),
    "n_values": int(vol.shape[0]),
    "first_timestamp": vol.index[0].strftime(fmt),
    "first_is_nan": bool(pd.isna(vol.iloc[0])),
    "mean_excluding_nan": float(vol.dropna().mean()),
    "samples": [{"position": int(i), "timestamp": vol.index[i].strftime(fmt),
                 "value": None if pd.isna(vol.iloc[i]) else float(vol.iloc[i])} for i in picks],
}
(HERE / "daily_vol_reference.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
