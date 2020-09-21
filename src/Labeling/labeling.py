import numpy as np
import pandas as pd
from utils import mp_pandas


def triple_barrier_method(close, events, pt_sl, molecule):
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method

    Applies triple-barrier labeling method on time-series (molecule).

    Returns DataFrame of timestamps of barrier touches.

    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Event values calculated (CUSUM filter)
    :param pt_sl: (np.array) Profit takin value 0; Stop loss value 1
    :param molecule: (an array) Datetime index values

    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['target']
    else:
        pt = pd.Series(index=events.index) # NaNs

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['target']
    else:
        sl = pd.Series(index=events.index) # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # path returns

        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min() # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking

    return out


def get_events(close, t_events, pt_sl, target, min_ret=0,
               num_threads=1, t1=False, side=None):
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Computes the time of first touch using Meta Labels.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) of t_events. The timestamps are calculated using the CUSUM filter
        and will be used as timestamps for the Triple Barrier Method.
    :param pt_sl: (2 element array) Profit takin value 0; Stop loss value 1.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) as a
        scalar values to calculate the size of the profit take and stop loss.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param t1: (pd.Series) A pandas series with the timestamps of the vertical barriers.
        Pass False to disable vertical barriers.
    :param side: (pd.Series) Long or Short side prediction.

    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['target'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] is profit taking multiple
            -events['sl']  is stop loss multiple
    """
    # Get sampled target values
    target = target.loc[t_events]
    target = target[target > min_ret]

    # Get time boundary t1
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Define the side
    if side is None:
        _side = pd.Series(1., index=target.index)
        _pt_sl = [pt_sl, pt_sl]
    else:
        _side = side.loc[target.index]
        _pt_sl = pt_sl[:2]

    events = pd.concat({'t1': t1, 'target': target, 'side': _side}, axis = 1)
    events = events.dropna(subset = ['target'])
    df0 = mp_pandas(func=triple_barrier_method, pd=('molecule', events.index),
                    num_threads=num_threads, close=close, events=events, pt_sl=pt_sl)
    events['t1'] = df0.dropna(how='all').min(axis=1) # ignores NaN

    if side is None:
        events = events.drop('side', axis=1)

    return events


def add_vertical_barrier(t_events, close, num_days=1):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.

    Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier

    :return: (pd.Series) Timestamps of vertical barriers
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1<close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=t_events[:t1.shape[0]]))

    return t1


def get_bins(events, close):
    """
    Advances in Financial Machine Learning, Snippet 3.7, page 51.

    Computes labels for side and size of the bets.

    :param events: (pd.DataFrame)
                -events.index is event's start time
                -events['t1'] is event's endtime
                -events['target'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (pd.Series) Close prices

    :return: (pd.DataFrame) Meta-labeled events
    """
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    px = events.index.union(events['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # Create out object
    out = pd.DataFrame(index = events.index)
    out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index] - 1

    if 'side' in events:
        out['ret'] *= events['side']

    out['bin'] = np.sign(out['ret'])

    if 'side' in events:
        out.loc[out['ret'] <= 0, 'bin'] = 0

    return out


def drop_labels(events, min_pct):
    """
    Advances in Financial Machine Learning, Snippet 3.8 pg. 64.

    Drops labels that are underpopulated

    :param events: (pd.Series) Value of events calculated (symmetric CUSUM filter)
    :min_pct: (float) Drops the lowest min_pct of the labels

    :return: (pd.Series) Events with the bottom min_pct labels removed
    """
    while True:
        df = events['bin'].value_counts(normalize=True)

        if df.min() > min_pct or df.shape[0] < 3:
            break

        print('dropped label', df.argmin(), df.min())

        events = events[events['bin'] != df.argmin()]

    return events
