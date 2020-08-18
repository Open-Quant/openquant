"""
This file stores the base class that will be used to create the various types of bars.

"""

import pandas as pd
import numpy as np


class BaseBars():
    """
    This base class will be used slack
    """
    def __init__(self, metric: str, batch_size: int = 2e7):
        """
        Constructor

        param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        param batch_size: (int) Number of rows to read in from the csv, per batch.
        """

        pass
    
    def batch_run(self, file_path: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True, 
                  to_csv: bool = False, output_path: Optional[str] = None) ->  Union[pd.DataFrame, None]:
        """
        Reads csv file or pd.DataFrame in batches and returns a pd.DataFrame. 
        Requirements: DataFrame must only contain three columns: timestamp, price, and volume.
        
        param file_path: (str) Path to csv file(s) or pd.DataFrame containing raw tick data 
        param verbose: (bool) Whether to print message on each processed batch
        param to_csv: (bool) Writing the results of bars generated to a local csv file or to in-memory pd.DataFrame
        param output_path: (bool) Path to results file, if to_csv = True 
        
        return: (pd.DataFrame or None) Financial data Structure
        """
        
        pass