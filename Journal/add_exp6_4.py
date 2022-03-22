# Distributionally Robust End-to-End Portfolio Construction
# Experiment 6 - Non-linear models
####################################################################################################
# Import libraries
####################################################################################################
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.close("all")

# Make the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import E2E_DRO functions
from e2edro import e2edro as e2e
from e2edro import DataLoad as dl
from e2edro import BaseModels as bm
from e2edro import PlotFunctions as pf

# Path to cache the data, models and results
cache_path = "./cache/new_exp6/"

####################################################################################################
# Load data
####################################################################################################
# Train, validation and test split percentage
split = [0.7, 0.3]

# Number of feattures and assets
n_x, n_y = 5, 10

# Number of observations per window and total number of observations
n_obs, n_tot = 100, 1200

# Synthetic data: randomly generate data from a linear model
X, Y = dl.synthetic_nl(n_x=n_x, n_y=n_y, n_obs=n_obs, n_tot=n_tot, split=split)

####################################################################################################
# EXPERIMENT 5: Data-rich vs data-starved
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Initialize parameters
#---------------------------------------------------------------------------------------------------

# Performance loss function and performance period 'v+1'
perf_loss='sharpe_loss'
perf_period = 13

# Weight assigned to MSE prediction loss function
pred_loss_factor = 0.5

# Risk function (default set to variance)
prisk = 'p_var'

# Robust decision layer to use: hellinger or tv
dr_layer = 'hellinger'

# Determine whether to train the prediction weights Theta
train_pred = True

# List of learning rates to test
lr_list = [0.005, 0.0125, 0.02]

# List of total no. of epochs to test
epoch_list = [20, 30, 40, 50]

# For replicability, set the random seed for the numerical experiments
set_seed = 6000

#---------------------------------------------------------------------------------------------------
# Run 
#---------------------------------------------------------------------------------------------------
# 4. DR E2E (3-layer)
nom_net_linear = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, 
                    set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor).double()
nom_net_linear.net_cv(X, Y, lr_list, epoch_list, n_val=1)
nom_net_linear.net_roll_test(X, Y, n_roll=1)
with open(cache_path+'nom_net_linear.pkl', 'wb') as outp:
    pickle.dump(nom_net_linear, outp, pickle.HIGHEST_PROTOCOL)
print('nom_net_linear run complete')