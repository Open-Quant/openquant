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
        pass
    else:
        print('error')


def dollar_bar_df_test(dataframe):
    """
    This validates that the dollar_bars_df() function returns a correct amount of columns 
    """
    df0 = standard_bars.dollar_bars_df(dataframe, 'cum_dollar', .05)
    print(df0)

    if len(df0) > 0
        pass
    else:
        print('error')


<<<<<<< HEAD
def volume_bars_test(dataframe):
    """
    This validates the volume_bars() function does not error
    """

    list0 = standard_bars.volume_bars(dataframe, 'cum_vol', .05)

    if len(list0) > 0:
        pass
    else:
        print('error')

def volume_bars_df_test(dataframe):
    """
    This validates the volume_bars_df() returns the correct amount of columns
    """
    df0 = standard_bars.volume_bars_df(dataframe, 'cum_vol', .05)
    print(df0)

    if len(df0) > 0:
        pass
    else:
        print('error')

def tick_bars_test(dataframe):
    """
    This validates tick_bars() function does not errorb 
    """
    list0 = standard_bars.tick_bars(dataframe, 'cum_ticks', .05)
    
    print(list0)
    
    if len(list0) > 0:
        pass
    else:
        print('error')
        
=======
def tick_bars_test(dataframe):
    """
    This validates that the tick_bar() function works
    """
    list0 = standard_bars.tick_bars(dataframe, 'close', 33)
    print('ok')


def tick_bar_df_test(dataframe):
    """
    This validates that the tick_bar() function works
    """
    df = standard_bars.tick_bar_df(dataframe, 'close', 33)
    print('ok')
>>>>>>> ab59ded59a1f4cb3dec68d53bfed0a70341dcde3

def tick_bars_df_test(dataframe):
    """
    This validates the tick_bars_df() returns the correct amount of columns
    """
   
    df0 = standard_bars.tick_bars_df(dataframe, 'cum_ticks', .05)
    print(df0)
    
    if len(df0) > 0:
        pass
    else:
        print('error')

def volume_bars_test(dataframe):
    """
    This validates that the volume_bar() function works
    """
    list0 = standard_bars.volume_bars(dataframe, 'close', 33)
    print('ok')


def volume_bar_df_test(dataframe):
    """
    This validates that the volume_bar() function works
    """
    df = standard_bars.volume_bar_df(dataframe, 'close', 33)
    print('ok')




# Below here is the code that runs the tests
df = pd.read_csv('../data/raw/raw_data.csv')

dollar_bars_test(df)
<<<<<<< HEAD
dollar_bars_df_test(df)
volume_bars_test(df)
volume_bars_df_test(df)
=======
dollar_bar_df_test(df)
tick_bars_test(df)
tick_bar_df_test(df)
volume_bars_test(df)
volume_bar_df_test(df)
>>>>>>> ab59ded59a1f4cb3dec68d53bfed0a70341dcde3

print('all tests complete')


