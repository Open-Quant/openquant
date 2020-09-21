import numpy as np
import pandas as pd 


def get_indicator_matrix(bar_index, t1):
    """
    Advances in Financial Machine Learning, Snippet 4.3, page 65.

    Builds an Indicator Matrix

    Get indicator matrix. 

    :param bar_index: (pd.Series): Triple barrier events(t1) from labeling.get_events
    :param t1: (pd.DataFrame): Price bars which were used to form triple barrier events
    :return: (np.array) Indicator binary matrix indicating what (price) bars influence the label for each observation
    """
    indicator_matrix = pd.DataFrame(0, index = bar_index, columns = range(t1.shape[0]))

    for i, (t0, t1) in enumerate(t1.iteritems()):
        indicator_matrix.loc[t0:t1, i] = 1

    return indicator_matrix

def get_average_uniqueness(indicator_matrix):
    """
    Advances in Financial Machine Learning, Snippet 4.4. page 65.

    Compute Average Uniqueness

    Average uniqueness from indicator matrix

    :param indicator_matrix: (np.matrix) Indicator binary matrix
    :return: (float) Average uniqueness
    """
    c = indicator_matrix.sum(axis = 1) #concurrency
    u = indicator_matrix.div(c,axis = 0) #uniqueness
    average_uniqueness = u[u > 0].mean()
    average_uniqueness = average_uniqueness.fillna(0) #average uniqueness

    return average_uniqueness

def sequential_bootstrap(indicator_matrix, sample_length = None):
    """
    Advances in Financial Machine Learning, Snippet 4.5, Snippet 4.6, page 65.

    Return Sample from Sequential Bootstrap

    Generate a sample via sequential bootstrap.

    :param ind_mat: (pd.DataFrame) Indicator matrix from triple barrier events
    :param sample_length: (int) Length of bootstrapped sample
    :return: (array) Bootstrapped samples indexes
    """
    if sample_length is None:
        sample_length = indicator_matrix.shape[1]

    phi = []

    while len(phi) < sample_length:
        c = indicator_matrix[phi].sum(axis = 1) + 1
        average_uniqueness = get_average_uniqueness(indicator_matrix, c)
        prob = (average_uniqueness / average_uniqueness.sum()).values
        phi += [np.random.choice(indicator_matrix.columns, p = prob)]

    return phi

def axu_monte_carlo():
    """
    Parallelized auxiliary function
    """
    pass
    return