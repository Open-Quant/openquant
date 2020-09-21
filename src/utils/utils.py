import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt
import sys

def get_daily_vol(close, lookback_period = 100):
    """
    Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    The daily volatility (calculated at intraday estimation points) equal to a the lookback of
    an exponentially weighted moving standard deviation. The function sets a dynamic
    threshold for stop-loss and take-profit orders based on volatility.

    :param close: (pd.Series) Closing prices
    :param lookback_period: (int) Lookback period to compute volatility

    :return: (pd.Series) Daily volatility value
    """
    # Daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days = 1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index = close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.iloc[df0.values].values - 1 # daily returns

    df0 = df0.ewm(span = lookback_period).std()

    return df0

def lin_parts(num_atoms, num_threads):
    """
    Advances in Financial Machine Learning snippet 20.5, page 306

    Forms molecules by partitioning  a list of atoms in equal sized subsets (single loop)
    """
    parts = np.linspace(0, num_atoms,min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)

    return parts


def nested_parts(num_atoms, num_threads, upper_triang = False):
    """
    Advances in Financial Machine Learning snippet 20.6, page 308

    Partition of atoms with inner loop
    """
    parts ,num_threads_ = [0], min(num_threads, num_atoms)

    for num in range(num_threads_):
        part = 1 + 4 (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)

    parts = np.round(parts).astype(int)

    if upper_triang: # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)

    return parts

def mp_pandas(func, pd_obj, num_threads = 24, mp_batches = 1, lin_mols = True, **kwargs):
    """
    Parallelize jobs, return a dataframe or series

    :param func: Function to be parallelized. Returns a DataFrame
    :pd_obj[0]: Name of argument used to pass the molecule
    :pd_obj[1]: List of atoms that will be grouped into molecules
    :kwargs: Any other argument needed by func

    :return: computes function in parallel

    """
    #if linMols: parts = lin_parts(len(argList[1]), num_threads * mp_batches)
    #else: parts = nested_parts(len(argList[1]), num_threads * mp_batches)
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []

    for i in range(1, len(parts)):
        job = {pd_obj[0]:pd_obj[1][parts[i - 1]:parts[i]], 'func':func}
        job.update(kwargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs,num_threads = num_threads)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out

    for i in out:
        df0 = df0.append(i)

    df0 = df0.sort_index()

    return df0

# TODO: Add in documentation
def process_jobs_(jobs):
    """
    Runs jobs sequentially for debugging
    """
    out=[]

    for job in jobs:
        out_=expand_call(job)
        out.append(out_)

    return out

# TODO: Add in documentation
def report_progress(job_num, num_jobs, time0, task):
    """
    Advances in Financial Machine Learning. Snippet 20.9 , page 312.

    Reports progress as async jobs are completed
    """
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))

    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after '\
        + str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')

    return

# TODO: Add in documentation
def process_jobs(jobs, task = None, num_threads = 24):
    """
    Advances in Financial Machine Learning. Snippet 20.9 , page 312.
    """
    if task is None:
        task = jobs[0]['func'].__name__

    pool = mp.Pool(processes = num_threads)
    outputs = pool.imap_unordered(expand_call, jobs)
    out = []
    time0 = time.time()

    # Process async output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        report_progress(i, len(jobs), time0, task)

    pool.close();pool.join() # this is needed to prevent memory leaks

    return out

# TODO: Add in documentation
def expand_call(kwargs):
    """
    Advances in Financial Machine Learning. Snippet 20.10 , page 312.
    """
    # Expand the arguments of a callback function, kwargs['func']
    func = kwargs['func']

    del kwargs['func']

    out = func(**kwargs)

    return out

# TODO: Add in documentation
def _pickle_method(method):
    """
    Advances in Financial Machine Learning. Snippet 20.11 , page 313.
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class

    return _unpickle_method, (func_name,obj,cls)

# TODO: Add in documentation
def _unpickle_method(func_name, obj, cls):
    """
    Advances in Financial Machine Learning. Snippet 20.11 , page 313.
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break

    return func.__get__(obj,cls)