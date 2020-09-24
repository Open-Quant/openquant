"""
This will test the functions in the financial_data_structures/standard_bars.py file
"""

import sys
import os
import inspect
import pandas as pd

# Hacky stuff that I want to replace with real idiomatic pyhon code as soon as possible. (packages)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from financial_data_structures import standard_bars


def dollar_bars_test(dataframe):
    """
    This validates that the dollar_bars() function does not error
    """
    list0 = standard_bars.dollar_bars(dataframe, 'cum_dollar', 70000000.75)
    if len(list0) > 0:
        print('dollar_bars_test() pass')
    else:
        print('error with dollar_bars_test()')


def dollar_bars_df_test(dataframe):
    """
    This validates that the dollar_bars_df() function returns a correct amount of columns 
    """
    df0 = standard_bars.dollar_bars_df(dataframe, 'cum_dollar', .05)
    if len(df0) > 0:
        print('dollar_bars_df_test() pass')
    else:
        print('error with dollar_bars_df_test()')


def tick_bars_test(dataframe):
    """
    This validates that the tick_bar() function works
    """
    list0 = standard_bars.tick_bars(dataframe, 'close', 33)
    if len(list0) > 0:
        print('tick_bars_test() pass')
    else:
        print('error with tick_vars_test()')


def tick_bars_df_test(dataframe):
    """
    This validates that the tick_bar() function works
    """
    df = standard_bars.tick_bars_df(dataframe, 'close', 33)
    if len(df.columns) > 0:
        print('tick_bars_df_test() ok')
    else:
        print('error with tick_bars_df_test()')


def volume_bars_test(dataframe):
    """
    This validates that the volume_bar() function works
    """
    list0 = standard_bars.volume_bars(dataframe, 'close', 33)
    if len(list0) > 0:
        print('volume_bars_test() pass')
    else:
        print('error with volume_bars_test()')


def volume_bars_df_test(dataframe):
    """
    This validates that the volume_bar() function works
    """
    df = standard_bars.volume_bars_df(dataframe, 'close', 33)
    if len(df.columns) > 0:
        print('volume_bars_df_test() pass')
    else:
        print('error with volume_bars_df_test()')




# RUN THE TESTS
df = pd.read_csv('../data/raw/raw_data.csv')

dollar_bars_test(df)
dollar_bars_df_test(df)

volume_bars_test(df)
volume_bars_df_test(df)

tick_bars_test(df)
tick_bars_df_test(df)

print('all tests complete')


