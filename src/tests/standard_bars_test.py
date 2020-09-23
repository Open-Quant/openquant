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
    print(list0)
    if len(list0) > 0:
        pass
    else:
        print('error')


def dollar_bars_df_test(dataframe):
    """
    This validates that the dollar_bars_df() function returns a correct ammount of columns
    """
    desired_length = len(dataframe.columns) + 1
    df = standard_bars.dollar_bar_df(dataframe, 'close', 33)
    if len(df.columns) == desired_length:
        pass
    else:
        print('error')





# standard_bars.dollar_bar_df()


# standard_bars.tick_bar_df()


# standard_bars.volume_bar_df()

# Below here is the code that runs the tests
df = pd.read_csv('../data/raw/raw_data.csv')
dollar_bars_test(df)
dollar_bars_df_test(df)
print('all tests complete')


