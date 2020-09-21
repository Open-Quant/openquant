import numpy as np
import pandas as pd

def get_daily_vol(close, lookback_period=100):
    """
    Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    The daily volatility (calculated at intraday estimation points) equal to a the lookback of
    an exponentially weighted moving standard deviation. The function sets a dynamic
    threshold for stop-loss and take-profit orders based on volatility.

    :param close: (pd.Series) Closing prices
    :param lookback_period: (int) Lookback period to compute volatility

    :return: (pd.Series) Daily volatility value
    """
    # Daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.iloc[df0.values].values - 1 # daily returns

    df0 = df0.ewm(span=lookback_period).std()

    return df0