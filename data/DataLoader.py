# DataLoader module
# 
# Loads data from Kenneth R. French's database OR generates syntehtic data
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import pickle
import torch
from torch.autograd import Variable
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time

data_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/saved_models/"

####################################################################################################
# TrainValTest class
####################################################################################################
class TrainValTest:
    def __init__(self, data, split):
        """Object to hold the training, validation and testing datasets

        Inputs
        data: pandas dataframe with time series data
        split: list of ratios that control the partition of data into training, testing and 
        validation sets. 
    
        Output. TrainValTest object with fields and functions:
        data: Field. Holds the original pandas dataframe
        train(): Function. Returns a pandas dataframe with the training subset of observations
        """
        self.data = data

        n_obs = self.data.shape[0]
        split = n_obs * np.cumsum(split)
        self.split = [int(i) for i in split]

    def split_update(self, split):
        """Update the list outlining the split ratio of training, validation and testing
        """
        n_obs = self.data.shape[0]
        split = n_obs * np.cumsum(split)
        self.split = [int(i) for i in split]

    def train(self):
        """Return the training subset of observations from the full dataset
        """
        return self.data[:self.split[0]]

    def val(self):
        """Return the validation subset of observations from the full dataset
        """
        return self.data[self.split[0]:self.split[1]]

    def test(self):
        """Return the testing subset of observations from the full dataset
        """
        return self.data[self.split[1]:]

####################################################################################################
# Generate synthetic data
####################################################################################################
def synthetic(n_x, n_y, n_tot, split):
    """Generates synthetic (normally-distributed) asset and factor data

    Inputs
    n_x: number of features
    n_y: number of assets
    n_tot: number of observations in the whole dataset
    split: train-validation-test split as percentages 

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    split = int(n_tot * split)

    # 'True' prediction bias and weights
    a = torch.rand(n_x)
    b = torch.randn(n_y, n_x)

    # Syntehtic features
    X = torch.randn(n_tot, n_y)

    # Synthetic outputs
    Y = a + X @ b + 0.3 * torch.randn(n_tot, n_x)

    # Convert them to Variable type for use with torch library
    X, Y = Variable(X), Variable(Y)
    
    # Partition dataset into training and testing sets
    X, Y = TrainValTest(), TrainValTest()
    X.train, X.val, X.test = X[:split[0]], X[split[0]:split[1]], X[split[1]:]
    Y.train, Y.val, Y.test = Y[:split[0]], Y[split[0]:split[1]], Y[split[1]:]

    return X, Y

#-----------------------------------------------------------------------------------------------
# Option 2: Load data from Kenneth French's data library 
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
#-----------------------------------------------------------------------------------------------
def FamaFrench(start, end, split, freq=''):
    """Load data from Kenneth French's data library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 

    Inputs
    start: start date
    end: end date
    split: train-validation-test split as percentages 
    freq: data frequency (daily, weekly, monthly)

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """

    if freq == 'daily' or freq == '_daily':

        freq = '_daily'

        # Download asset data
        Y = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, 
                    end=end)[0] / 100

        # Download factor data
        X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                    end=end)[0]
        rf_df = X['RF']
        X = X.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

    elif freq == 'weekly' or freq == '_weekly':

        freq = '_daily'

        # Download asset data
        Y = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, 
                    end=end)[0] / 100

        # Download factor data
        X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                    end=end)[0]
        rf_df = X['RF']
        X = X.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

        # Convert daily returns to weekly returns
        Y = Y.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)
        X = X.resample('W-THU').agg(lambda x: (x + 1).prod() - 1)

    elif freq == 'monthly' or freq == '_monthly' or freq == '':

        freq = ''

        # Download asset data
        Y = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, 
                    end=end)[0] / 100

        # Download factor data
        X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                    end=end)[0]
        rf_df = X['RF']
        X = X.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

    # Partition dataset into training and testing sets
    X, Y = TrainValTest(X, split), TrainValTest(Y, split)

    return X, Y

#-----------------------------------------------------------------------------------------------
# Option 3: Factors from Kenneth French's data library and asset data from AlphaVantage
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
# https://www.alphavantage.co 
#-----------------------------------------------------------------------------------------------
def AV(start, end, split, freq='weekly', use_cache=False):
    """Load data from Kenneth French's data library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 

    Inputs
    start: start date
    end: end date
    split: train-validation-test split as percentages 
    freq: data frequency (daily, weekly, monthly)

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """

    data_path = '/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/saved_models/asset_factor_'+freq+'.pkl'

    if use_cache:
        with open(data_path, 'rb') as inp:
            X = pickle.load(inp)
            Y = pickle.load(inp)
    else:
        ts = TimeSeries(key='CV2O4TLLVRI8TGMD', output_format='pandas', indexing_type='date')

        if freq == 'daily' or freq == '_daily':

            freq = '_daily'

            # Download asset data
            tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT', 'JNJ', 'PFE']
            Y = []
            for tick in tick_list:
                data, _ = ts.get_daily_adjusted(symbol=tick, outputsize='full')
                data = data['5. adjusted close']
                Y.append(data)
                time.sleep(12.5)
            Y = pd.concat(Y, axis=1)
            Y = Y['1999-1-1':end].pct_change()
            Y = Y[start:end]
            Y.columns = tick_list

            # Download factor data
            X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                        end=end)[0]
            rf_df = X['RF']
            X = X.drop(['RF'], axis=1)
            mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
            st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
            lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

            # Concatenate factors as a pandas dataframe
            X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

        elif freq == 'weekly' or freq == '_weekly':

            freq = '_daily'

            # Download asset data
            tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT', 'JNJ', 'PFE']
            Y = []
            for tick in tick_list:
                data, _ = ts.get_daily_adjusted(symbol=tick, outputsize='full')
                data = data['5. adjusted close']
                Y.append(data)
                time.sleep(12.5)
            Y = pd.concat(Y, axis=1)
            Y = Y['1999-1-1':end].pct_change()
            Y = Y[start:end]
            Y.columns = tick_list

            # Download factor data 
            X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                        end=end)[0]
            rf_df = X['RF']
            X = X.drop(['RF'], axis=1)
            mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
            st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
            lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

            # Concatenate factors as a pandas dataframe
            X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

            # Convert daily returns to weekly returns
            Y = Y.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)
            X = X.resample('W-THU').agg(lambda x: (x + 1).prod() - 1)

        # Partition dataset into training and testing sets
        X, Y = TrainValTest(X, split), TrainValTest(Y, split)

        with open(data_path, 'wb') as outp:
            pickle.dump(X, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(Y, outp, pickle.HIGHEST_PROTOCOL)
            X, Y

    return X, Y