# File written by Sean and Soren on 6/28/2020
# Code mostly from https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
# The plan with this code is to turn each of the secions into separate methods for our data class.
# That way, we can specify the format we want to send to S3 (command-line-arguments), and then it will compute and upload.

# Imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import helper_functions as hf

# Initialize file and dataframe
inputFile = sys.argv[1]
print("Input file is: ", inputFile)
data = pd.read_csv(inputFile)

# Parse and remove 0s. This is very specific to the example we are following. 
# We don't need to do this for our purposes.
data['timestamp'] = data.timestamp.map(lambda t:
    datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))


# 1) Create TIME bars
data_timeidx = data.set_index('timestamp')
data_time_grp = data_timeidx.groupby(pd.Grouper(freq="15Min"))
num_time_bars = len(data_time_grp)
data_time_vwap = data_time_grp.apply(hf.compute_vwap)
print(data_time_vwap)


# 2) Create TICK bars
total_ticks = len(data) # Using Tick data in example
num_ticks_per_bar = total_ticks / num_time_bars
num_ticks_per_bar = round(num_ticks_per_bar, -3) # three places (arb)
data_tick_grp = data.reset_index().assign(grpId=lambda row: row.index // num_ticks_per_bar)
data_tick_vwap = data_tick_grp.groupby('grpId').apply(hf.compute_vwap)
data_tick_vwap.set_index('timestamp',  inplace=True)
print(data_tick_vwap)


# 3) Create VOLUME bars
data_cm_vol = data.assign(cmVol=data['homeNotional'].cumsum())
total_vol = data_cm_vol.cmVol.values[-1] # what is this transformation...?
vol_per_bar = total_vol / num_time_bars
vol_per_bar = round(vol_per_bar, -2) # round to the nearest hundred
data_vol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)
data_vol_vwap = data_vol_grp.groupby('grpId').apply(hf.compute_vwap)
data_vol_vwap.set_index('timestamp', inplace=True)
print(data_vol_vwap)


# 4) Create DOLLAR bars
data_cm_vol = data.assign(cmVol=data['foreignNotional'].cumsum())
total_vol = data_cm_vol.cmVol.values[-1] # what is this transformation...?
vol_per_bar = total_vol / num_time_bars
vol_per_bar = round(vol_per_bar, -2) # round to the nearest hundred
data_vol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)
data_vol_vwap = data_vol_grp.groupby('grpId').apply(hf.compute_vwap)
data_vol_vwap.set_index('timestamp', inplace=True)
print(data_vol_vwap)


# 5) Create TICK IMBALANCE bars
# Adds tick direction to data
data_timeidx['tickDirection'] = data_timeidx.tickDirection.map(hf.convert_tick_direction) 
# Compute signed flow of each tick
data_signed_flow = data_timeidx.assign(bv = data_timeidx.tickDirection * data_timeidx.size)
# Accumulate Dollar Inbalance Bars
abs_Ebv_init = np.abs(data_signed_flow['bv'].mean())
E_T_init = 500000 # 500000 ticks to warm up
Ts, abs_thetas, thresholds, i_s = hf.compute_Ts(data_signed_flow.bv, E_T_init, abs_Ebv_init)


### THIS IS A WAY TO USE THE COMMAND LINE TO CREATE A NEW DATASET TRANSFORMED (TICK, DOLLAR, VOLUME)