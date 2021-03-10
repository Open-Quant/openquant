# File written by Sean and Soren on 5 July 2020
# With help from:  https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Labelling/Trend-Follow-Question.ipynb
# NOTE for future iterations
# This could be modified by adding command line arguments to serve as the variables for the threshold for the function calls. 
# If we wanted, we could make ONE call and send every type of bar to an S3 bucket if we added that functionality.


# Imports...
import mlfinlab as ml
from mlfinlab.datasets import (load_dollar_bar_sample)
import mlfinlab.data_structures as ds
import numpy as np
import pandas as pd
import pyfolio as pf
import timeit
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import argparse

# Set our command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-csv', '--csv_name', help='The name of the csv file. Must be inside of ml-pipeline folder')
parser.add_argument('-th', '--threshold', help='Set the threshold for your data.')
parser.add_argument('-bs', '--batch_size', help='The size of each batch you will create.')
parser.add_argument('-npb', '--num_previous_bars', help='The number of previous bars.')
parser.add_argument('-eiw', '--expected_imbalance_window', help='The expected window of imbalance')
parser.add_argument('-enti', '--exp_num_ticks_init', help='The expected number of expected ticks.')
parser.add_argument('-entc', '--exp_num_ticks_constraints', help='The expected number of tick constraints.')

# Assign the command line args to variables
args = parser.parse_args()
csv_name = args.csv_name
threshold = args.threshold
batch_size = args.batch_size
num_previous_bars = args.num_previous_bars
expected_imbalance_window = args.expected_imbalance_window
exp_num_ticks_init = args.exp_num_ticks_init
exp_num_ticks_constraints = args.exp_num_ticks_constraints

# Logic to validate tags and set defaults if not supplied
if csv_name == 'None':
    sys.exit('Must supply a csv name with -csv tag!')

# Check if threshold was supplied
if threshold == None:
    threshold = 28000
else:
    threshold = int(threshold)

# Check if batch_size was supplied
if batch_size == None:
    batch_size = 10000000
else:
    batch_size = int(batch_size)

# Check if num_previous_bars was supplied
if num_previous_bars == None:
    num_previous_bars = 3
else:
    num_previous_bars = int(num_previous_bars)

# Check if expected_imbalance_window was supplied
if expected_imbalance_window == None:
    expected_imbalance_window = 10000
else:
    expected_imbalance_window = int(expected_imbalance_window)

# Check if exp_num_ticks_init was supplied
if exp_num_ticks_init == None:
    exp_num_ticks_init = 2000
else:
    exp_num_ticks_init = int(exp_num_ticks_init)

# Check if exp_num_ticks_constraints was supplied
if exp_num_ticks_constraints == None:
    exp_num_ticks_constraints = [0, 100]
else:
    exp_num_ticks_constraints = exp_num_ticks_constraints.split(',')

# Ouput all of the arguments
print('-------------------------')
print('A R G U M E N T S')
print(' ')
print('csv_name: {}'.format(csv_name))
print('threshold: {}'.format(threshold))
print('batch_size: {}'.format(batch_size))
print('num_previous_bars: {}'.format(num_previous_bars))
print('expected_imbalance_window: {}'.format(expected_imbalance_window))
print('exp_num_ticks_init: {}'.format(exp_num_ticks_init))
print('exp_num_ticks_constraints: {}'.format(exp_num_ticks_constraints))
print('-------------------------')

# Set up dataframe
data = pd.read_csv(csv_name)
#data = data.drop(
#    ['low', 'open', 'close', 'period_volume', 'number_trades'], axis=1)
#data.columns = ['date_time', 'price', 'total_volume']

# CREATING STANDARD BARS

# Create dollar bar
print('Creating Dollar Bars')
dollar_bar = ds.get_dollar_bars(
    data, threshold=28000, batch_size=10000000, verbose=True)

# Create volume bar
print('Creating Volume Bars')
volume_bar = ds.get_volume_bars(
    data, threshold=28000, batch_size=1000000, verbose=True)

# Create tick bar
print('Creating Tick Bars')
tick_bar = ds.get_tick_bars(data, threshold=28000,
                            batch_size=1000000, verbose=True)


# CREATING EMA IMBALANCE BARS

# Create EMA Dollar Imbalance bar
print('Creating EMA Dollar Imbalance Bar')
ema_dollar_imbalance_bar = ds.get_ema_dollar_imbalance_bars(data, num_prev_bars=3, expected_imbalance_window=10000, exp_num_ticks_init=2000,
                                                            exp_num_ticks_constraints=[0, 100], batch_size=100, verbose=True, to_csv=True, output_path='dollar-imbalance.csv')

# Create EMA Tick Imbalance bar
print('Creating EMA Tick Imbalance Bar')
ema_tick_imbalance_bar = ds.get_ema_tick_imbalance_bars(data, num_prev_bars=3, expected_imbalance_window=10000, exp_num_ticks_init=2000,
                                                        exp_num_ticks_constraints=[0, 100], batch_size=100, verbose=True)

# Create EMA Volume Imbalance bar
print('Creating EMA Volume Imbalance Bar')
ema_volume_imbalance_bar = ds.get_ema_volume_imbalance_bars(data, num_prev_bars=3, expected_imbalance_window=10000, exp_num_ticks_init=2000,
                                                            exp_num_ticks_constraints=[0, 100], batch_size=100, verbose=True)

# CREATING CONSTANT IMBALANCE BARS

# Create Constant Dollar Imbalance Bar
print('Creating Constant Dollar Imbalance Bar')
constant_dollar_imbalance_bar = ds.get_const_dollar_imbalance_bars(data, expected_imbalance_window=1000, exp_num_ticks_init=2000,
                                                                   batch_size=100, verbose=True)

# Create Constant Volume Imbalance Bar
print('Creating Constant Volume Imbalance Bar')
constant_dollar_imbalance_bar = ds.get_const_volume_imbalance_bars(data, expected_imbalance_window=1000, exp_num_ticks_init=2000,
                                                                   batch_size=100, verbose=True)

# Create Constant Tick Imbalance Bar
print('Creating Constant Tick Imbalance Bar')
constant_tick_imbalance_bar = ds.get_const_tick_imbalance_bars(data, expected_imbalance_window=1000, exp_num_ticks_init=2000,
                                                               batch_size=100, verbose=True)

# CREATING EMA RUN BARS

# Create EMA Dollar Run Bar
print('Creating EMA Dollar Run Bar')
ema_dollar_run_bar = ds.get_ema_dollar_run_bars(data, num_prev_bars=3, expected_imbalance_window=1000, exp_num_ticks_init=5,
                                                exp_num_ticks_constraints=[0, 100], batch_size=100, verbose=True)

# Create EMA Volume Run Bar
print('Creating EMA Volume Run Bar')
ema_volume_run_bar = ds.get_ema_volume_run_bars(data, num_prev_bars=3, expected_imbalance_window=1000,
                                                exp_num_ticks_init=5, exp_num_ticks_constraints=[0, 100], batch_size=100, verbose=True)

# Create EMA Tick Run Bar
print('Creating EMA Tick Run Bar')
ema_tick_run_bar = ds.get_ema_tick_run_bars(data, num_prev_bars=3, expected_imbalance_window=1000,
                                            exp_num_ticks_init=5, exp_num_ticks_constraints=[0, 100], batch_size=100, verbose=True)


# CREATING CONSTANT RUN BARS

# Create Constant Dollar Bar
print('Creating Constant Dollar Run Bar')
const_dollar_run_bar = ds.get_const_dollar_run_bars(data, num_prev_bars=3, expected_imbalance_window=1000,
                                                   exp_num_ticks_init=5, batch_size=100, verbose=True)

# Create Constant Volume Bar
print('Creating Constant Volume Run Bar')
const_volume_run_bar = ds.get_const_volume_run_bars(data, num_prev_bars=3, expected_imbalance_window=1000,
                                                exp_num_ticks_init=5, batch_size=100, verbose=True)

# Create Constant Tick Bar
print('Creating Constant Tick Run Bar')
const_tick_run_bar = ds.get_const_tick_run_bars(data, num_prev_bars=3, expected_imbalance_window=1000,
                                            exp_num_ticks_init=5, batch_size=100, verbose=True)

print('BARS CREATED')

# could add each dataframe to vector, and then loop through list, and upload each to aws.
