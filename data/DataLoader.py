# DataLoader module
# 
# Loads data from Kenneth R. French's database OR generates syntehtic data
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
from torch.autograd import Variable
import pandas as pd
import pandas_datareader as pdr
import numpy as np

####################################################################################################
# TrainValTest class
####################################################################################################
class TrainValTest:

    def __init__(self):

        self.train = []
        self.val = []
        self.test = []

####################################################################################################
# Generate synthetic data
####################################################################################################
def synthetic(n_x, n_y, n_tot, split):
    """Function that generates synthetic (normally-distributed) data

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
    X_data = torch.randn(n_tot, n_y)

    # Synthetic outputs
    Y_data = a + X_data @ b + 0.3 * torch.randn(n_tot, n_x)

    # Convert them to Variable type for use with torch library
    X_data, Y_data = Variable(X_data), Variable(Y_data)
    
    # Partition dataset into training and testing sets
    X, Y = TrainValTest(), TrainValTest()
    X.train, X.val, X.test = X_data[:split[0]], X_data[split[0]:split[1]], X_data[split[1]:]
    Y.train, Y.val, Y.test = Y_data[:split[0]], Y_data[split[0]:split[1]], Y_data[split[1]:]

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

        # Download data 
        Y_data = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, 
                    end=end)[0] / 100
        X_data = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                    end=end)[0]
        rf_df = X_data['RF']
        X_data = X_data.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X_data = pd.concat([X_data, mom_df, st_df, lt_df], axis=1) / 100

    elif freq == 'weekly' or freq == '_weekly':

        freq = '_daily'

        # Download data 
        Y_data = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, 
                    end=end)[0] / 100
        X_data = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                    end=end)[0]
        rf_df = X_data['RF']
        X_data = X_data.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X_data = pd.concat([X_data, mom_df, st_df, lt_df], axis=1) / 100

        # Convert daily returns to weekly returns
        Y_data = Y_data.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)
        X_data = X_data.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)

    elif freq == 'monthly' or freq == '_monthly' or freq == '':

        freq = ''

        # Download data 
        Y_data = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, 
                    end=end)[0] / 100
        X_data = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start,
                    end=end)[0]
        rf_df = X_data['RF']
        X_data = X_data.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X_data = pd.concat([X_data, mom_df, st_df, lt_df], axis=1) / 100

    n_obs = X_data.shape[0]
    split = n_obs * np.cumsum(split)
    split = [int(i) for i in split]

    # Partition dataset into training and testing sets
    X, Y = TrainValTest(), TrainValTest()
    X.train, X.val, X.test = X_data[:split[0]], X_data[split[0]:split[1]], X_data[split[1]:]
    Y.train, Y.val, Y.test = Y_data[:split[0]], Y_data[split[0]:split[1]], Y_data[split[1]:]

    return X, Y