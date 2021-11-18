# End-to-End Distributionally Robust Optimization
#
# Prepared by:    Giorgio Costa (gc2958@columbia.edu)
# Last revision:  31-Oct-2021
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pandas_datareader as pdr

# Import E2E_DRO functions
my_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/E2E-DRO"
import sys
sys.path.append(my_path)
from e2edro import e2edro as dro
from e2edro import e2enom as nom
from e2edro import LossFunctions as lf
from e2edro import RiskFunctions as rf

# Imoprt 'reload' to update E2E_DRO libraries while in development
from importlib import reload 
reload(dro)
reload(nom)

import matplotlib.pyplot as plt
plt.close("all")

####################################################################################################
# Load data
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Option 1: Generate synthetic data
#---------------------------------------------------------------------------------------------------
torch.manual_seed(1)
# Number of observations T, features m and outputs n
T, m, n = 112, 8, 15

# 'True' prediction bias and weights
a = torch.rand(n)
b = torch.randn(m,n)

# Syntehtic features
X = torch.randn(T, m)

# Synthetic outputs
Y_test = a + X @ b + 0.2*torch.randn(T,n)

# Convert them to Variable type for use with torch library
X, Y = Variable(X), Variable(Y_test)

T = 100
X, Y = X[0:T], Y[0:T]

#---------------------------------------------------------------------------------------------------
# Option 2: Load data from Kenneth French's data library 
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
#---------------------------------------------------------------------------------------------------
# Data frequency: type '_daily' for daily frequency, or '' for monthly frequency
freq = ''
start = '1980-01-01'
end = '2021-09-30'

# Download data 
asset_df = pdr.get_data_famafrench('10_Industry_Portfolios'+freq, start=start, end=end)[0]
factor_df = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+freq, start=start, end=end)[0]
rf_df = factor_df['RF']
factor_df = factor_df.drop(['RF'], axis=1)
mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+freq, start=start, end=end)[0]
st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+freq, start=start, end=end)[0]
lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+freq, start=start, end=end)[0]

# Concatenate factors as a pandas dataframe
factor_df = pd.concat([factor_df, mom_df, st_df, lt_df], axis=1)

# Convert pd.dataframes to torch.variables and trim the dataset
X = Variable(torch.tensor(factor_df.values / 100, dtype=torch.double))[0:400,:]
Y = Variable(torch.tensor(asset_df.values / 100, dtype=torch.double))[0:400,0:10]

# Partition dataset into training and testing sets
X_train, X_val, X_test = X[:150], X[150:250], X[250:]
Y_train, Y_val, Y_test = Y[:150], Y[150:250], Y[250:]

# Declare number of features n_x, number of assets n_y and number of observations n_obs
# n_obs is the number of observations given to the NN for distributional analysis
n_x, n_y, n_obs = X.shape[1], Y.shape[1], 80

####################################################################################################
# Neural net training and testing
####################################################################################################
# Evaluation metrics
perf_loss=lf.single_period_over_var_loss
pred_loss_factor = 0.1
epochs = 3
lr = 0.025

#---------------------------------------------------------------------------------------------------
# E2E Nominal neural net
#---------------------------------------------------------------------------------------------------
# Neural net object
nom_net = dro.e2e(n_x, n_y, n_obs, prisk=rf.p_var).double()

# Train and validate neural net
nom_results = nom_net.net_train(X_train, Y_train, X_val, Y_val, epochs=epochs, lr = lr, 
                perf_loss=perf_loss, pred_loss_factor=pred_loss_factor)

# Ouf-of-sample test
nom_p = nom_net.net_test(X_test, Y_test)

#---------------------------------------------------------------------------------------------------
# E2E DRO neural net
#---------------------------------------------------------------------------------------------------
# Neural net object
dro_net = dro.e2e(n_x, n_y, n_obs, prisk=rf.p_var, opt_layer='hellinger').double()

# Train and validate neural net
dro_results = dro_net.net_train(X_train, Y_train, X_val, Y_val, epochs=epochs, lr = lr,
                perf_loss=perf_loss, pred_loss_factor=pred_loss_factor)

# Ouf-of-sample test
dro_p = dro_net.net_test(X_test, Y_test)

# Print parameter values and gradients
for name, param in dro_net.named_parameters():
    print(name, param.grad.data)
    print(name, param.data)
    
# Save/load trained model
model_path = my_path+"/saved_models/dro_net"
torch.save(dro_net, model_path)
test = torch.load(model_path)




nom_results.plot()


plt.show()