"""
This will test the functions in the financial_data_structures/standard_bars.py file
"""
import sys
import os
import inspect
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from financial_data_structures import standard_bars


def standard_bars_test(dataframe):
    df0 = standard_bars.dollar_bar_df(dataframe, 0, 5)
    if df0.columns > 0:
        pass
    else:
        print('error')


df = pd.read_csv('../data/raw/raw_data.csv')
print(df)

standard_bars_test(df)

# standard_bars.dollar_bar_df()


# standard_bars.tick_bar_df()


# standard_bars.volume_bar_df()

print('all tests complete')


