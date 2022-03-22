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
# Generate linear synthetic data
####################################################################################################
def synthetic(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=100):
    """Generates synthetic (normally-distributed) asset and factor data

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 250) + 0.0001
    b = np.random.randn(n_x, n_y) / 5
    c = np.random.randn(int((n_x+1)/2), n_y)

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_nl(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=100):
    """Generates synthetic (normally-distributed) factor data and mix them following a quadratic 
    model of linear, squared and cross products to produce the asset data. 

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 200) + 0.0005
    b = np.random.randn(n_x, n_y) / 4
    c = np.random.randn(int((n_x+1)/2), n_y)
    d = np.random.randn(n_x**2, n_y) / n_x

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50
    X_cross = 100 * (X[:,:,None] * X[:,None,:]).reshape(n_tot, n_x**2)
    X_cross = X_cross - X_cross.mean(axis=0)

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + X_cross @ d + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

#-----------------------------------------------------------------------------------------------
# Option 3: Factors from Kenneth French's data library and asset data from AlphaVantage
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
# https://www.alphavantage.co 
#-----------------------------------------------------------------------------------------------
def AV(start, end, split, freq='weekly', n_obs=104, n_y=None, use_cache=False, save_results=False, 
        AV_key=None):
    """Load data from Kenneth French's data library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 

    Inputs
    start: start date
    end: end date
    split: train-validation-test split as percentages 
    freq: data frequency (daily, weekly, monthly)
    n_obs: number of observations per batch
    use_cache: Boolean. State whether to load cached data or download data
    save_results: Boolean. State whether the data should be cached for future use. 

    Outputs
    X: TrainTest object with feature data split into train, validation and test subsets
    Y: TrainTest object with asset data split into train, validation and test subsets
    """

    if use_cache:
        X = pd.read_pickle('./cache/factor_'+freq+'.pkl')
        Y = pd.read_pickle('./cache/asset_'+freq+'.pkl')
    else:
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT', 'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']

        if n_y is not None:
            tick_list = tick_list[:n_y]

        if AV_key is None:
            print("""A personal AlphaVantage API key is required to load the asset pricing data. If you do not have a key, you can get one from www.alphavantage.co (free for academic users)""")
            AV_key = input("Enter your AlphaVantage API key: ")

        ts = TimeSeries(key=AV_key, output_format='pandas', indexing_type='date')

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

        if save_results:
            X.to_pickle('./cache/factor_'+freq+'.pkl')
            Y.to_pickle('./cache/asset_'+freq+'.pkl')

    # Partition dataset into training and testing sets. Lag the data by one observation
    return TrainTest(X[:-1], n_obs, split), TrainTest(Y[1:], n_obs, split)