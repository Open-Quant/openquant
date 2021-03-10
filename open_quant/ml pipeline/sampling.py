# File written by Sean and Soren on July 13, 2020
# File written in reference to url: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Sample%20Weights/Sequential_Bootstrapping.ipynb
# The data to be used for this file is generated from triple_barrier.py. That script will output a csv called 'barrier_events.csv'.
# The dollar bar csv is available here: # Data from here: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Labelling/sample_dollar_bars.csv

# The purpose of sampling label uniqueness is to make the data more stationary by removing some of ambiguety
# created when there are parralel labels impacting price.  Here we are sampling to boost higher average uniquess of our dataset

# Imports
import seaborn as sns
import mlfinlab as ml
from numba import jit, prange
from mlfinlab.sampling.bootstrapping import get_ind_mat_average_uniqueness, get_ind_matrix, seq_bootstrap
from mlfinlab.sampling.concurrent import get_av_uniqueness_from_triple_barrier
import pandas as pd
import numpy as np

# Get barrier events (you might have to drop duplicate timestamps...)
barrier_events = pd.read_csv('barrier_events.csv', parse_dates=[0])
barrier_events.drop_duplicates(subset="t1", keep=False, inplace=True)
barrier_events.set_index('t1', drop=False, inplace=True)

# Get our close prices from csv
close_prices = pd.read_csv('stupid_data.csv', index_col=0, parse_dates=[0,2])
print(close_prices)

# We can measure average label uniqueness using get_av_uniqueness_from_tripple_barrier function from mlfinlab package
av_unique = get_av_uniqueness_from_triple_barrier(barrier_events, close_prices.close, num_threads=3)
av_unique.mean()
print(av_unique.mean())

# Index of the first unique label
unique_label_index = av_unique[av_unique.tW==1].index[0] # take the first sample
print(unique_label_index)

barrier_events[barrier_events.index >= unique_label_index].head()  ### Figure out why this does not work

### Bagging, Bootstrapping and Random Forrest

# Ensemble learning technique (bagging with replacement) the goal is to randomly choose data samples
# that are unique and non-concurrent for each decision tree
# With sequential bootsrapping our goal is to select samples such that with each iteration we can
# maximize average unqiueness of subsamples

ind_mat = pd.DataFrame(index=range(0,6), columns=range(0,3))
ind_mat.loc[:, 0] = [1, 1, 1, 0, 0, 0]
ind_mat.loc[:, 1] = [0, 0, 1, 1, 0, 0]
ind_mat.loc[:, 2] = [0, 0, 0, 0, 1, 1]
ind_mat
print(ind_mat)

# Get triple barier method indicator matrix
triple_barrier_ind_mat = get_ind_matrix(barrier_events, price_bars=close_prices['close'])
print(triple_barrier_ind_mat)

ind_mat_uniqueness = get_ind_mat_average_uniqueness(triple_barrier_ind_mat)  ### CHECK BACK AFTER FIXING DUPLICATE T Values
print(ind_mat_uniqueness)

first_sample = ind_mat_uniqueness
first_sample[first_sample > 0].mean()

# Jupyter notebook output
# av_unique.loc[0]

# Get the values
ind_mat = ind_mat.values

# On the first step all labels will have equal probabilities as average uniquess of matrix with 1 column is 1
phi = [1]
uniqueness_array = np.array([None, None, None])
for i in range(0,3):
    ind_mat_reduced = ind_mat[:, phi + [i]]
    label_uniqueness = get_ind_mat_average_uniqueness(ind_mat_reduced)#[-1] # The last value corresponds to appended i TODO fix this
    uniqueness_array[i] = (label_uniqueness[label_uniqueness > 0].mean())
prob_array = uniqueness_array / sum(uniqueness_array)

print(prob_array)
phi = [1,2]
unqiueness_array = np.array([None, None, None])
for i in range(0,3):
    ind_mat_reduced = ind_mat[:, phi + [i]]
    label_uniqueness = get_ind_mat_average_uniqueness(ind_mat_reduced)#[-1] TODO fix this
    uniqueness_array[i] = (label_uniqueness[label_uniqueness > 0].mean())
prob_array = uniqueness_array / sum(uniqueness_array)

phi = [1, 2, 0]
uniqueness_array = np.array([None, None, None])
for i in range(0, 3):
    ind_mat_reduced = ind_mat[:, phi + [i]]
    label_uniqueness = get_ind_mat_average_uniqueness(ind_mat_reduced)#[-1] TODO fix thiss
    uniqueness_array[i] = (label_uniqueness[label_uniqueness > 0].mean())
prob_array = uniqueness_array / sum(uniqueness_array)

print(prob_array)

samples = seq_bootstrap(ind_mat, sample_length=4, warmup_samples=[1], verbose=True)
print(samples)

### Monte-Carlo experiment (checks to see how sequential bootsrapping will improve average label uniqueness)

standard_unq_array = np.zeros(10000) * np.nan # Array of random sampling uniqueness
seq_unq_array = np.zeros(10000) * np.nan # Array of Sequential Bootstapping uniqueness
for i in range(0, 10000):
    bootstrapped_samples = seq_bootstrap(ind_mat, sample_length=3)
    random_samples = np.random.choice(ind_mat.shape[1], size=3)

    random_unq = get_ind_mat_average_uniqueness(ind_mat[:, random_samples])
    random_unq_mean = random_unq[random_unq > 0].mean()

    sequential_unq = get_ind_mat_average_uniqueness(ind_mat[:, bootstrapped_samples])
    sequential_unq_mean = sequential_unq[sequential_unq > 0].mean()

    standard_unq_array[i] = random_unq_mean
    seq_unq_array[i] = sequential_unq_mean

np.median(standard_unq_array), np.median(seq_unq_array)
np.mean(standard_unq_array), np.mean(seq_unq_array)

# KDE plots of label uniqueness support the fact taht sequential bootstrapping gives higher average label uniqueness
sns.kdeplot(standard_unq_array, shade=True, label='Random Sampling')
sns.kdeplot(seq_unq_array, shade=True, label='Sequential Sampling')

# Let's apply sequential bootstrapping to our full data set and draw 50 samples.
bootstrapped_samples = seq_bootstrap(triple_barrier_ind_mat, compare=True, sample_length=50)
print(bootstrapped_samples)


### A BUNCH OF PLOTTING (PLOTLY) CODE
plt.show()
plt.axis()
plt.y_label()
plt.x_label()
