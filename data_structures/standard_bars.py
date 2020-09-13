# The purpose of this file is to store the functions that generate standard bars.
# Standard bars are a way of recreating the chonological time bars so as to better reflect the inflow and outflow of (dollars, volume etc) TODO try and make this easier to understand. 

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def compute_vwap(df):
    q = df['foreingNotional']
    p = df['price']
    vwap = np.sum(p * q) / np.sum(q)
    df['vwap'] = vwap
    return df

data_timeidx = data.set_index('timestamp')
data_time_grp = data_timeidx.groupby(pd.Grouper(freq='15Min'))
num_time_bars = len(data_time_grp) # comes in handy later
data_time_vwap = data_time_grp.apply(compute_vwap)

# TODO write docs
def dollar_bars():
    
    print('dollar bars')

# TODO write docs
def tick_bars():
    print('tick bars')

# TODO write docs
def volume_vars():
    data_cm_vol = data.assign(cmVol=data['homeNotional'].cumsum()) 
    total_vol = data_cm_vol.cmVol.values[-1]
    vol_per_bar = total_vol / num_time_bars
    vol_per_bar = round(vol_per_bar, -2) # round to the nearest hundred
    
    data_vol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)
    
    data_vol_vwap =  data_vol_grp.groupby('grpId').apply(compute_vwap)
    data_vol_vwap.set_index('timestamp', inplace=True)
    print('volume bars')

print('hello world')