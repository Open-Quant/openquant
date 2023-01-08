import pandas as pd
import numpy as np
import scipy.stats
import math


class RiskMetrics:
    """
    This class contains methods for calculating common risk metrics used in trading and asset management.
    """

    def __init__(self):
        return

    @staticmethod
    def calculate_variance(covariance, weights):
        """
        Calculate the variance of a portfolio/asset.

        :param covariance: (pd.DataFrame/np.matrix) covariance matrix of assets
        :param weights: (list) list of asset weights

        :return: (float) variance
        """

        return np.dot(weights, np.dot(covariance, weights))

    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.05):
        """
        Calculate the value at risk (VaR) of a portfolio/asset. The amount
        of money that is at risk over a given period of time given 
        a confidence interval.

        :param returns: (pd.DataFrame/np.array) asset/portfolio historical returns
        :param confidence_level: (float) confidence level (alpha)

        :return: (float) VaR
        """

        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        return returns.quantile(confidence_level, interpolation='higher')[0]

    def calculate_expected_shortfall(self, returns, confidence_level=0.05):
        """
        Calculate the expected shortfall (CVaR) of a portfolio/asset. This 
        is used to measure risk in a worst case scenario (tail event).

        :param returns: (pd.DataFrame/np.array) asset/portfolio historical returns
        :param confidence_level: (float) confidence level (alpha)

        :return: (float) expected shortfall
        """

        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        value_at_risk = self.calculate_value_at_risk(returns, confidence_level)
        expected_shortfall = np.nanmean(returns[returns < value_at_risk])
        return expected_shortfall

    @staticmethod
    def calculate_conditional_drawdown_risk(returns, confidence_level=0.05):
        """
        Calculate the conditional drawdown of risk (CDaR) of a portfolio/asset.

        :param returns: (pd.DataFrame/np.array) asset/portfolio historical returns
        :param confidence_level: (float) confidence level (alpha)

        :return: (float) conditional drawdown risk
        """

        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)

        drawdown = returns.expanding().max() - returns
        max_drawdown = drawdown.expanding().max()
        max_drawdown_at_confidence_level = max_drawdown.quantile(confidence_level, interpolation='higher')
        conditional_drawdown = np.nanmean(max_drawdown[max_drawdown > max_drawdown_at_confidence_level])
        return conditional_drawdown
    
    @staticmethod
    def skewness(r):
        """
        Computes the skewness of the supplied Series or DataFrame.

        :param r: (pd.Series or pd.DataFrame) Asset returns

        :return: (pd.Series or Float) Skewness of r
        """
        demeaned_r = r - r.mean()

        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()

        return exp / sigma_r**3

    @staticmethod
    def kurtosis(r):
        """
        Computes Kurtosis. Kurtosis greater >3 is leptokurtic and suggests
        significant level of tail risk.

        :param r: (pd.Series or pd.DataFrame) Asset returns

        :return: (pd.Series or float) Kurtosis of r
        """
        demeaned_r = r - r.mean()

        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()

        return exp / sigma_r**4
