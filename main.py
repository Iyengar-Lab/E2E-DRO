# End-to-End Distributionally Robust Optimization
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
# %% Import libraries
####################################################################################################
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

# Import E2E_DRO functions
my_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL"

model_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/saved_models/"

import sys
sys.path.append(my_path+"/E2E-DRO")
from e2edro import e2edro as dro
from e2edro import e2edro2 as dro2
from e2edro import DataLoad as dl
from other import PlotFunctions as pf
from other import NaiveModel as nm

# Make the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Imoprt 'reload' to update E2E_DRO libraries while in development
from importlib import reload 
reload(dro)
reload(dro2)
reload(dl)
reload(pf)
reload(nm)

####################################################################################################
# %% Load data
####################################################################################################
# Data frequency and start/end dates
freq = 'weekly'
start = '2000-01-01'
end = '2021-09-30'

# Train, validation and test split percentage
split = [0.6, 0.4]

# Number of observations per window 
n_obs = 104

# Number of assets
n_y = 20

# Load data
X, Y = dl.AV(start, end, split, freq=freq, n_obs=n_obs, use_cache=True, n_y=n_y)

# Number of features and assets
n_x, n_y = X.data.shape[1], Y.data.shape[1]

####################################################################################################
# %% Neural net training and testing
####################################################################################################
# Evaluation metrics
perf_loss='sharpe_loss'
perf_period = 13
pred_loss_factor = 0.5
prisk = 'p_var'
opt_layer = 'hellinger'
train_pred = True
set_seed = 1

lr_list = [0.01, 0.02, 0.03]
epoch_list = [10, 20, 30]
use_cache = False

#---------------------------------------------------------------------------------------------------
# %% Run neural nets
#---------------------------------------------------------------------------------------------------
if use_cache:
    with open(model_path+'nom_dro_nets_'+prisk+'_TrainPred'+str(train_pred)+'.pkl', 'rb') as inp:
        nom_net = pickle.load(inp)
        dro_net = pickle.load(inp)
        naive_net = pickle.load(inp)

else:
    # Nominal E2E neural net
    nom_net = dro2.e2e(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, set_seed=set_seed,
                        perf_loss=perf_loss, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor).double()
    init_gamma = nom_net.gamma.item()
    nom_net.net_cv(X, Y, lr_list, epoch_list)
    nom_net.net_roll_test(X, Y, n_roll=4)

    # DRO E2E DRO neural net
    dro_net = dro2.e2e(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, set_seed=set_seed,
                        opt_layer=opt_layer, perf_loss=perf_loss, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor).double()
    dro_net.net_cv(X, Y, lr_list, epoch_list)
    dro_net.net_roll_test(X, Y, n_roll=4)

    # Naive predict-then-optimize model
    naive_net = nm.pred_then_opt(n_x, n_y, n_obs, gamma=init_gamma, prisk=prisk).double()
    naive_net.net_roll_test(X, Y, n_roll=4)

    with open(model_path+'nom_dro_nets_'+prisk+'_TrainPred'+str(train_pred)+'.pkl', 'wb') as outp:
        pickle.dump(nom_net, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dro_net, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(naive_net, outp, pickle.HIGHEST_PROTOCOL)

####################################################################################################
# %% Plots
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Wealth evolution plot
#---------------------------------------------------------------------------------------------------
pf.wealth_plot(nom_net.portfolio, dro_net.portfolio, naive_net.portfolio, path=my_path+"/plots/wealth.pdf")
