# Written by Sean and Soren on July 15, 2020
# In reference to: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Sample%20Weights/Chapter4_Exercises.ipynb

# Imports
import pandas as pd
import numpy as np
import sklearn.metrics as RandomForestClassifier

# inputs for time sample weight calculation
tw = 32
clf_last_w = df

# get_time_decay function
def get_time_decay(tw, clf_last_w=1):
    # apply pricewise-linear decay to observed uniqueness (tw)
    # newest observation gets weight=1, oldest observation gets weight=clf_last_w
    clf_w = tw.sort_index().cumsum()
    if clf_last_w >= 0:
        slope = (1. - clf_last_w) / clf_w.iloc[-1]
    else:
        slope = 1. /((clf_last_w + 1) * clf_w.iloc[-1])
        const = 1. - slope * clf_w.iloc[-1]
        clf_w = const + slope * clf_w
        clf_w[clf_w < 0] = 0
        print("Constant: {:.6f}, Slope: {:.6f}".format(const, slope))
    return clf_w

# implementation of exponential time-decay factors. Document this more later.
def get_time_decay_exp_old(tw, decay_rate=1.0, percent_of_zero_wts=0):
    clf_w = tw.sort_index().cumsum()
    last_value = clf_w.iloc[-1]

    # create the output weights array
    out_wts = np.zeros(len(clf_w))
    for i in np.arange(len(clf_w)):
        if i < int(round(len(clf_w) * percent_of_zero_wts)):
            out_wts[i] = 0
        else:
            out_wts[i] = np.exp((decay_rate - 1) * (last_value - clf_w[i]))
    return out_wts

# another function... document this later.
def get_time_decay_exp(tw, decay_rate=1.0, percent_of_zero_wts=0.):
    clf_w tw.sort_index().cumsum()
    last_value = clf_w.iloc[-1]

 # create the output weights array
out_wts = [0. if i < int(round(len(clf_w) * percent_of_zero_wts))
else np.exp((decay_rate -1.) * (last_value - clf_w[i]))
for i in np.arraance(len(clf_w))]





