# End-to-End Distributionally Robust Optimization
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
# %% Import libraries
####################################################################################################
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
from data import DataLoader as dl
from other import PlotFunctions as pf
from other import NaiveModel as nm

# Make the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Imoprt 'reload' to update E2E_DRO libraries while in development
from importlib import reload 
reload(dro)
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
split = [0.5, 0.2, 0.3]

# Load data
n_y = 6
X, Y = dl.AV(start, end, split, freq=freq, use_cache=True, n_y=n_y)

# Number of features and assets
n_x, n_y = X.data.shape[1], Y.data.shape[1]

# Number of observations per window 
n_obs = 104

####################################################################################################
# %% Neural net training and testing
####################################################################################################
# Evaluation metrics
perf_loss='sharpe_loss'
perf_period = 10
pred_loss_factor = 0.2
epochs = 15
prisk = 'p_var'
opt_layer = 'hellinger'

# Pretrain OLS weights
pre_params = nm.pred_then_opt(n_x, n_y, n_obs, prisk=prisk, gamma=0.2)
pre_params.train(X, Y)
pre_params = pre_params.Theta.T

#---------------------------------------------------------------------------------------------------
# %% Naive model
#---------------------------------------------------------------------------------------------------
reload(nm)

naive_net = nm.pred_then_opt(n_x, n_y, n_obs, prisk=prisk, gamma=0.1)
naive_net.train(X, Y)

naive_p = naive_net.test(X, Y)

#---------------------------------------------------------------------------------------------------
# %% E2E Nominal neural net
#---------------------------------------------------------------------------------------------------
# Set learning rate
lr = 0.01
turnover = False

# split2 = [0.7, 0, 0.3]
# X.split_update(split2)
# Y.split_update(split2)

# Neural net object (use CUDA if available)
nom_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, turnover=turnover).double()
nom_net = nom_net.to(device)


# Train and validate neural net
nom_results = nom_net.net_train(X, Y, epochs=epochs, lr=lr, perf_loss=perf_loss,
                perf_period=perf_period, pred_loss_factor=pred_loss_factor, pre_params=pre_params)

# Save dataframe with training results
# nom_results.to_pickle(my_path+"/saved_models/nom_results.pkl")
# nom_results = pd.read_pickle(my_path+"/saved_models/nom_results.pkl")

# Ouf-of-sample test
# nom_net.load_state_dict(torch.load(my_path+"/saved_models/nom_net"))
nom_p = nom_net.net_test(X, Y)

# Ouf-of-sample test of 'best trained' model
nom_net_best = dro.e2e(n_x, n_y, n_obs, prisk=prisk, turnover=turnover).double()
nom_net_best.load_state_dict(torch.load(my_path+"/saved_models/nom_net_best"))
nom_p_best = nom_net_best.net_test(X, Y)

#---------------------------------------------------------------------------------------------------
# %% E2E DRO neural net
#---------------------------------------------------------------------------------------------------
# Set learning rate
lr = 0.02
turnover = False

# Neural net object (use CUDA if available)
dro_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer, turnover=turnover).double()
dro_net = dro_net.to(device)

# Train and validate neural net
dro_results = dro_net.net_train(X, Y, epochs=epochs, lr=lr, perf_loss=perf_loss, 
                perf_period=perf_period, pred_loss_factor=pred_loss_factor, pre_params=pre_params)

# Save dataframe with training results
# dro_results.to_pickle(my_path+"/saved_models/dro_results.pkl")
# dro_results = pd.read_pickle(my_path+"/saved_models/dro_results.pkl")

# Ouf-of-sample test
# dro_net.load_state_dict(torch.load(my_path+"/saved_models/dro_net"))
dro_p = dro_net.net_test(X, Y)

# Ouf-of-sample test of 'best trained' model
dro_net_best = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer,turnover=turnover).double()
dro_net_best.load_state_dict(torch.load(my_path+"/saved_models/dro_net_best"))
dro_p_best = dro_net_best.net_test(X, Y)

#---------------------------------------------------------------------------------------------------
# %% Train multiple models
#---------------------------------------------------------------------------------------------------

for k in range(15,25):

    # Set learning rate
    lr = 0.001 + (0.001 * k)

    # Neural net object
    nom_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, iter_id=k).double()

    # Train and validate neural net
    nom_results = nom_net.net_train(X.train(), Y.train(), X.val(), Y.val(), epochs=epochs, lr=lr, 
                    perf_loss=perf_loss, pred_loss_factor=pred_loss_factor, iter_id=str(k))

    # Save dataframe with training results
    nom_results.to_pickle(my_path+"/saved_models/nom_results_"+str(k)+".pkl")

    # Neural net object
    dro_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer, iter_id=k).double()

    # Train and validate neural net
    dro_results = dro_net.net_train(X.train(), Y.train(), X.val(), Y.val(), epochs=epochs, lr=lr,
                    perf_loss=perf_loss, pred_loss_factor=pred_loss_factor, iter_id=str(k))

    # Save dataframe with training results
    dro_results.to_pickle(my_path+"/saved_models/dro_results_"+str(k)+".pkl")

#---------------------------------------------------------------------------------------------------
# %% Analyze results
#---------------------------------------------------------------------------------------------------
nom_results = []
dro_results = []

nom_p = []
nom_p_best = []

dro_p = []
dro_p_best = []

for k in range(15,25):

    nom_results.append(pd.read_pickle(my_path+"/saved_models/nom_results_"+str(k)+".pkl"))
    dro_results.append(pd.read_pickle(my_path+"/saved_models/dro_results_"+str(k)+".pkl"))

    nom_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, iter_id=k).double()
    nom_net.load_state_dict(torch.load(my_path+"/saved_models/nom_net_final"+str(k)))
    nom_p.append(nom_net.net_test(X, Y))

    nom_net_best = dro.e2e(n_x, n_y, n_obs, prisk=prisk, iter_id=k).double()
    nom_net_best.load_state_dict(torch.load(my_path+"/saved_models/nom_net_best"+str(k)))
    nom_p_best.append(nom_net_best.net_test(X, Y))

    dro_net = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer).double()
    dro_net.load_state_dict(torch.load(my_path+"/saved_models/dro_net_final"+str(k)))
    dro_p.append(dro_net.net_test(X, Y))

    dro_net_best = dro.e2e(n_x, n_y, n_obs, prisk=prisk, opt_layer=opt_layer).double()
    dro_net_best.load_state_dict(torch.load(my_path+"/saved_models/dro_net_best"+str(k)))
    dro_p_best.append(dro_net_best.net_test(X, Y))

nom_p_best_idx = np.zeros((10,3))
dro_p_best_idx = np.zeros((10,3))

for k in range(0,10):
    nom_p_best_idx[k,0] = nom_results[k].val_loss.iloc[-1]
    nom_p_best_idx[k,1] = nom_results[k].val_loss.idxmin()
    nom_p_best_idx[k,2] = nom_results[k].val_loss.min()

    dro_p_best_idx[k,0] = dro_results[k].val_loss.iloc[-1]
    dro_p_best_idx[k,1] = dro_results[k].val_loss.idxmin()
    dro_p_best_idx[k,2] = dro_results[k].val_loss.min()

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



reload(pf)
#---------------------------------------------------------------------------------------------------
# Loss plots
#---------------------------------------------------------------------------------------------------
pf.loss_plot_multiple(nom_results[:5], dro_results[:5], path=my_path+"/plots/loss_multi_1.pdf")
pf.loss_plot_multiple(nom_results[5:], dro_results[5:], path=my_path+"/plots/loss_multi_2.pdf")

pf.loss_plot_multiple(nom_results, dro_results, path=my_path+"/plots/loss_multi.pdf")

reload(pf)
#---------------------------------------------------------------------------------------------------
# Gamma and delta plots
#---------------------------------------------------------------------------------------------------
pf.gamma_plot_multiple(nom_results[:5], dro_results[:5], path=my_path+"/plots/gamma_multi_1.pdf")
pf.gamma_plot_multiple(nom_results[5:], dro_results[5:], path=my_path+"/plots/gamma_multi_2.pdf")

pf.gamma_plot_multiple(nom_results, dro_results, path=my_path+"/plots/gamma_multi.pdf")

reload(pf)
#---------------------------------------------------------------------------------------------------
# Wealth evolution plots
#---------------------------------------------------------------------------------------------------
pf.wealth_plot_multiple(nom_p_best[:5], dro_p_best[:5], path=my_path+"/plots/wealth_multi_1.pdf")
pf.wealth_plot_multiple(nom_p_best[5:], dro_p_best[5:], path=my_path+"/plots/wealth_multi_2.pdf")

pf.wealth_plot_multiple(nom_p_best, dro_p_best, path=my_path+"/plots/wealth_multi.pdf")

pf.wealth_plot_multiple(nom_p_best, dro_p_best)


# %%

# Data load module (including transformation to weekly data)
# Plotting module
# Tune nominal and DRO models
# Compare delta and gamma evolution

