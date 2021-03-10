### This file contains the bollinger bands related functions

def bbands(close_prices, window, no_of_stdev):
    """
    This function calculates bollinger bands.

    :param close_prices: (pd.DataFrame) Close prices of an asset
    :param window: The window for which to calculate the bollinger bands
    :param no_of_stdev: the number of standard deviations (used to determine the spread between the bollinger bands
    
    :return: Returns the upper, lower, and middle bands
    """
    rolling_mean = close_prices.ewm(span=window).mean()
    rolling_std = close_prices.ewm(span=window).std()

    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    return rolling_mean, upper_band, lower_band