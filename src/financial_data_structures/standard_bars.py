# The purpose of this file is to store the functions that generate standard bars.
# Standard bars are a way of recreating the chronological time bars so as to better reflect the inflow and outflow of (dollars, volume etc) TODO try and make this easier to understand. 

# TODO: Implement Bar creation class that is for standard bars (tick, volume, dollar)

import pandas as pd

def dollar_bars(df, dollar_value_column, threshold):
    """
    Computes dollar bars (index)

    :param df: pd.DataFrame
    :param dollar_value_column: (str) name of the dollar value column
    :param threshold: (int) threshold value for dollar bar creation

    :return idx: (list) list of tick bar indices
    """
    t = df[dollar_value_column]
    ts = 0
    idx = []

    for i, x in enumerate(t):
        ts += x
        if ts >= threshold:
            idx.append(i)
            ts = 0
    return idx


def dollar_bars_df(df, dollar_value_column, threshold):
    """
    Computes dollar bars

    :param df: pd.DataFrame()
    :param dollar_value_column: (str) name of the dollar value column
    :param threshold: (int) threshold value for dollar bar creation

    :return: pd.DataFrame() dollar bars
    """
    idx = dollar_bars(df, dollar_value_column, threshold)
    df['dollar_bar'] = idx
    df.drop_duplicates()

    return df


def tick_bars(df, price_column, threshold):
    """
    Computes tick bars (index)

    :param df: pd.DataFrame()
    :param price: (str) name of the price data column
    :param m: (int) threshold value for tick bar creation

    :return idx: (list) list of tick bar indices
    """
    t_price = df[price_column]
    ts = 0
    idx = []

    for i, x in enumerate(t_price):
        ts += 1
        if ts >= threshold:
            idx.append(i)
            ts = 0

    return idx


def tick_bars_df(df, price_column, threshold):
    """
    Computes tick bars (pd.Dataframe)

    :param df: pd.DataFrame()
    :param price_column: (str) name of the price data column

    :return: pd.DataFrame() tick bars
    """
    idx = tick_bars(df, price_column, threshold)

    return df.iloc[idx].drop_duplicates()


def volume_bars(df, volume_column, threshold):
    """
    Computes volume bars (index)

    :param df: pd.DataFrame()
    :param volume_column: (str) name for the column with volume data
    :param m: (int) threshold for volume

    :return idx: returns list of indices
    """
    t = df[volume_column]
    ts = 0
    idx = []

    for i, x in enumerate(t):
        ts += x
        if ts >= threshold:
            idx.append(i)
            ts = 0

    return idx


def volume_bars_df(df, volume_column, threshold):
    """
    Computes volume bars (pd.DataFrame)

    :param df: pd.DataFrame()
    :param volume_column: (str) name for the column with volume data
    :param m: (int) threshold for volume

    :return: pd.DataFrame() volume bars
    """
    idx = volume_bars(df, volume_column, threshold)

    return df.iloc[idx].drop_duplicates()