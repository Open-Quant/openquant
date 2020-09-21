import pandas as pd
import numpy as np
import scipy.stats
import math

def skewness(r):
    """
    Computes the skewness of the supplied Series or DataFrame

    :r: (pd.Series or pd.DataFrame) Asset returns

    :return: (pd.Series or Float) Skewness of r
    """
    demeaned_r = r - r.mean()

    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()

    return exp / sigma_r**3


def kurtosis(r):
    """
    Computes Kurtosis

    :param r: (pd.Series or pd.DataFrame) Asset returns

    :return: (pd.Series or float) Kurtosis of r
    """
    demeaned_r = r - r.mean()

    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()

    return exp / sigma_r**4