# This file was written by Soren and Sean in late June 2020.
# This file is the 'main' function that will drive the program.
# I will document the different command-line-arguments here once we get to that point.
# Imports
import os
import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import NoCredentialsError
import sys
from sklearn.preprocessing import OneHotEncoder
from data_class import DataFrame
import mlfinlab.data_structures as ds

# Initialize command line arguments
csv_name = sys.argv[1]

# Read csv into data frame
df = pd.read_csv(csv_name)

# Create an instance of our DataFrame class to modify data
data = DataFrame(df, csv_name)

# Transform data (columns we want- price, date_time, volume)
cols_to_keep = ['date_time', 'period_volume', 'high']
data.df = data.df[cols_to_keep]

# Dollar bar
dsb_filename = 'data_standard_bar.csv'
ds.get_dollar_bars(
    data.df,
    threshold=28000,
    batch_size=10000000,
    verbose=False,
    to_csv=True,
    output_path=dsb_filename)
frame = pd.read_csv(dsb_filename)
frame.to_csv(dsb_filename, index=False) # we did this because we wanted to remove the index

# Ema Dollar Imbalance Bar
deib_filename = 'data_ema_imbalance_bar.csv'
ds.get_ema_dollar_imbalance_bars(
    data.df,
    num_prev_bars=3,
    expected_imbalance_window=10000,
    exp_num_ticks_init=2000,
    exp_num_ticks_constraints=[0, 100],
    batch_size=100,
    verbose=True,
    to_csv=True,
    output_path=deib_filename)
frame = pd.read_csv(deib_filename)
frame.to_csv(deib_filename, index=False) # we did this becasue we wanted to remove the index

# Dumb print message. Remove this somemtime later.
print('Standard and EMA Imbalance Data created')
print(data.df)

# Upload to AWS
data.write_df_to_csv()
data.upload_to_aws(data.final_name)
data.upload_to_aws(dsb_filename)
data.upload_to_aws(deib_filename)

# Delete temporary data that we just uploaded to aws
os.remove(dsb_filename)
os.remove(deib_filename)



### ADD IN FEATURE TO SPLIT STRUCTURED DATA INTO TRAIN AND TEST SETS

