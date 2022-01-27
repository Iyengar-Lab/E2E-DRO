# DataLoad module
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
from torch.autograd import Variable
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time

####################################################################################################
# TrainTest class
####################################################################################################
class TrainTest:
    def __init__(self, data, n_obs, split):
        """Object to hold the training, validation and testing datasets

        Inputs
        data: pandas dataframe with time series data
        n_obs: Number of observations per batch
        split: list of ratios that control the partition of data into training, testing and 
        validation sets. 
    
        Output. TrainTest object with fields and functions:
        data: Field. Holds the original pandas dataframe
        train(): Function. Returns a pandas dataframe with the training subset of observations
        """
        self.data = data
        self.n_obs = n_obs
        self.split = split

        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def split_update(self, split):
        """Update the list outlining the split ratio of training, validation and testing
        """
        self.split = split
        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def train(self):
        """Return the training subset of observations
        """
        return self.data[:self.numel[0]]

    def test(self):
        """Return the test subset of observations
        """
        return self.data[self.numel[0]-self.n_obs:self.numel[1]]

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
    X, Y = TrainTest(), TrainTest()
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
    X: TrainTest object with feature data split into train, validation and test subsets
    Y: TrainTest object with asset data split into train, validation and test subsets
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
    X, Y = TrainTest(X, split), TrainTest(Y, split)

    return X, Y

#-----------------------------------------------------------------------------------------------
# Option 3: Factors from Kenneth French's data library and asset data from AlphaVantage
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
# https://www.alphavantage.co 
#-----------------------------------------------------------------------------------------------
def AV(start, end, split, freq='weekly', n_obs=104, use_cache=False, n_y=None):
    """Load data from Kenneth French's data library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 

    Inputs
    start: start date
    end: end date
    split: train-validation-test split as percentages 
    freq: data frequency (daily, weekly, monthly)

    Outputs
    X: TrainTest object with feature data split into train, validation and test subsets
    Y: TrainTest object with asset data split into train, validation and test subsets
    """

    if use_cache:
        X = pd.read_pickle('./saved_models/factor_'+freq+'.pkl')
        Y = pd.read_pickle('./saved_models/asset_'+freq+'.pkl')
    else:
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT', 'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']

        if n_y is not None:
            tick_list = tick_list[:n_y]

        ts = TimeSeries(key='CV2O4TLLVRI8TGMD', output_format='pandas', indexing_type='date')

        # Download asset data
        Y = []
        for tick in tick_list:
            data, _ = ts.get_daily_adjusted(symbol=tick, outputsize='full')
            data = data['5. adjusted close']
            Y.append(data)
            time.sleep(12.5)
        Y = pd.concat(Y, axis=1)
        Y = Y[::-1]
        Y = Y['1999-1-1':end].pct_change()
        Y = Y[start:end]
        Y.columns = tick_list

        # Download factor data 
        dl_freq = '_daily'
        X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+dl_freq, start=start,
                    end=end)[0]
        rf_df = X['RF']
        X = X.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+dl_freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+dl_freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+dl_freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

        if freq == 'weekly' or freq == '_weekly':
            # Convert daily returns to weekly returns
            Y = Y.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)
            X = X.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)

        X.to_pickle('./saved_models/factor_'+freq+'.pkl')
        Y.to_pickle('./saved_models/asset_'+freq+'.pkl')

    # Partition dataset into training and testing sets. Lag the data by one observation
    return TrainTest(X[:-1], n_obs, split), TrainTest(Y[1:], n_obs, split)