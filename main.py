# End-to-End Distributionally Robust Optimization
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
# %% Import libraries
####################################################################################################
import torch
import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")

# Import E2E_DRO functions
my_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL"
import sys
sys.path.append(my_path+"/E2E-DRO")
from e2edro import e2edro as dro
from data import DataLoader as dl
from other import PlotFunctions as pf
from other import NaiveModel as nm

# Imoprt 'reload' to update E2E_DRO libraries while in development
from importlib import reload 
reload(dro)
reload(dl)
reload(pf)

####################################################################################################
# %% Load data
####################################################################################################
# Data frequency and start/end dates
freq = 'weekly'
start = '2000-01-01'
end = '2021-09-30'

# Train, validation and test split percentage
split = [0.5, 0.2, 0.3]

# Load data
X, Y = dl.FamaFrench(start, end, split, freq)

# Number of features and assets
n_x, n_y = X.train.shape[1], Y.train.shape[1]

# Number of observations per window 
n_obs = 104

####################################################################################################
# %% Neural net training and testing
####################################################################################################
# Evaluation metrics
perf_loss='single_period_over_var_loss'
pred_loss_factor = 0.1
epochs = 75
prisk = 'p_var'
opt_layer = 'hellinger'

#---------------------------------------------------------------------------------------------------
# %% E2E Nominal neural net
#---------------------------------------------------------------------------------------------------
# Set learning rate
lr = 0.01

# Neural net object
nom_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk).double()

# Train and validate neural net
nom_results = nom_net.net_train(X.train, Y.train, X.val, Y.val, epochs=epochs, lr=lr, 
                perf_loss=perf_loss, pred_loss_factor=pred_loss_factor)

# Save dataframe with training results
# nom_results.to_pickle(my_path+"/saved_models/nom_results.pkl")
nom_results = pd.read_pickle(my_path+"/saved_models/nom_results.pkl")

# Ouf-of-sample test
# nom_net.load_state_dict(torch.load(my_path+"/saved_models/nom_net"))
nom_p = nom_net.net_test(X.test, Y.test)

# Ouf-of-sample test of 'best trained' model
nom_net_best = dro.e2e(n_x, n_y, n_obs).double()
nom_net_best.load_state_dict(torch.load(my_path+"/saved_models/nom_net_best"))
nom_p_best = nom_net_best.net_test(X.test, Y.test)

#---------------------------------------------------------------------------------------------------
# %% E2E DRO neural net
#---------------------------------------------------------------------------------------------------
# Set learning rate
lr = 0.025

# Neural net object
dro_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer).double()

# Train and validate neural net
dro_results = dro_net.net_train(X.train, Y.train, X.val, Y.val, epochs=epochs, lr=lr,
                perf_loss=perf_loss, pred_loss_factor=pred_loss_factor)

# Save dataframe with training results
# dro_results.to_pickle(my_path+"/saved_models/dro_results.pkl")
# dro_results = pd.read_pickle(my_path+"/saved_models/dro_results.pkl")

# Ouf-of-sample test
# dro_net.load_state_dict(torch.load(my_path+"/saved_models/dro_net"))
dro_p = dro_net.net_test(X.test, Y.test)

# Ouf-of-sample test of 'best trained' model
dro_net_best = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer).double()
dro_net_best.load_state_dict(torch.load(my_path+"/saved_models/dro_net_best"))
dro_p_best = dro_net_best.net_test(X.test, Y.test)

#---------------------------------------------------------------------------------------------------
# %% Naive model
#---------------------------------------------------------------------------------------------------
reload(nm)

naive_net = nm.pred_then_opt(n_x, n_y, n_obs, prisk=prisk, gamma=1.5)
naive_net.train(X.train, Y.train)

naive_p = naive_net.test(X.test, Y.test)

####################################################################################################
# %% Plots
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Loss plots
#---------------------------------------------------------------------------------------------------
pf.loss_plot(nom_results, path=my_path+"/plots/loss_nom.pdf")
pf.loss_plot(dro_results, path=my_path+"/plots/loss_dro.pdf")

#---------------------------------------------------------------------------------------------------
# Gamma and delta plots
#---------------------------------------------------------------------------------------------------
pf.gamma_plot(nom_results, path=my_path+"/plots/gamma_nom.pdf")
pf.gamma_plot(dro_results, path=my_path+"/plots/gamma_dro.pdf")

#---------------------------------------------------------------------------------------------------
# Wealth evolution plots
#---------------------------------------------------------------------------------------------------
pf.wealth_plot(nom_p_best, dro_p_best, path=my_path+"/plots/wealth.pdf")


# %%

# Data load module (including transformation to weekly data)
# Plotting module
# Tune nominal and DRO models
# Compare delta and gamma evolution