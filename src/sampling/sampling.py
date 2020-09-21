import pandas as pd

def number_of_concurrent_events(close_index, t1, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.1, page 60.

    Estimating the Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param close_index: (pd.Series) Close prices index
    :param t1: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param molecule: (an array) A set of datetime index values for processing

    :return: (pd.Series) Number concurrent labels for each datetime index
    """
    # find events that span the period [molocule[0],molocule[-1]]
    t1 = t1.fillna(close_index[-1]) # unclosed events still impact other weights
    t1 = t1[t1>molecule[0]] # events that end at or after molocule[0]
    t1 = t1.loc[:t1[molecule].max()] # events that start at or before t1[molocule.max()]

    # count events spanning a bar
    iloc = close_index.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index = close_index[iloc[0]:iloc[1] + 1])
    for t_in, t_out in t1.iteritems():
        count.loc[t_in:t_out] += 1

    return count.loc[molecule[0]:t1[molecule].max()]

def sample_average_uniqueness(t1, num_co_events, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.2, page 62.

    Estimating the Average Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param t1: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_co_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Average uniqueness over event's lifespan.
    """
    # derive average uniqueness over the events lifespan
    weight = pd.Series(index = molecule)

    for t_in, t_out in t1.loc[weight.index].iteritems():
        weight.loc[t_in] = (1. / num_co_events.loc[t_in: t_out]).mean()

    return weight