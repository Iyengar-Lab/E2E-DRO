# Distributionally Robust End-to-End Portfolio Construction
#
####################################################################################################
# %% Import libraries
####################################################################################################
from tkinter.tix import Tree
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
cache_path = "./cache/"

####################################################################################################
# %% Load data
####################################################################################################
# AlphaVantage API Key. 
# Note: User API keys can be obtained for free from www.alphavantage.co. Users will need a free 
# academic or paid license to donwload adjusted closing pricing data from AlphaVantage.
AV_key = None

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

# Download data (or load cached data)
X, Y = dl.AV(start, end, split, freq=freq, n_obs=n_obs, n_y=n_y, use_cache=True,
            save_results=False, AV_key=AV_key)

# Number of features and assets
n_x, n_y = X.data.shape[1], Y.data.shape[1]

####################################################################################################
# %% Neural net training and testing
####################################################################################################
# INITIALIZE PARAMETERS

# Performance loss function and performance period 'v+1'
perf_loss='sharpe_loss'
perf_period = 13

# Weight assigned to MSE prediction loss function
pred_loss_factor = 0.5

# Risk function (default set to variance)
prisk = 'p_var'

# Determine whether to train the prediction weights Theta (default is True for E2E models)
train_pred = True

# List of learning rates to test
lr_list = [0.01, 0.02, 0.03]

# List of total no. of epochs to test
epoch_list = [10, 20, 30]

# Load saved models (default is False)
use_cache = False

# Cache model results (default is False)
save_results = False

# For numerical experiments in manuscript, initialize gamma and delta to these values
init_params = [0.04288157820701599, 0.05563848838210106]

#---------------------------------------------------------------------------------------------------
# %% Run neural nets
#---------------------------------------------------------------------------------------------------

if use_cache:
    # Load cached models and backtest results
    with open(cache_path+'cached_models.pkl', 'rb') as inp:
        nom_net = pickle.load(inp)
        dr_net = pickle.load(inp)
        po_net = pickle.load(inp)
        base_net = pickle.load(inp)
        ew_net = pickle.load(inp)
        nom_net_const_gam = pickle.load(inp)
        dr_net_const_del = pickle.load(inp)

else:
    # Nominal E2E system
    nom_net = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                        init_params=init_params, opt_layer='nominal', perf_loss=perf_loss, 
                        perf_period=perf_period, pred_loss_factor=pred_loss_factor).double()
    nom_net.net_cv(X, Y, lr_list, epoch_list)
    nom_net.net_roll_test(X, Y, n_roll=4)

    # DR E2E system
    dr_net = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                        init_params=init_params, opt_layer='hellinger', perf_loss=perf_loss, 
                        perf_period=perf_period, pred_loss_factor=pred_loss_factor).double()
    dr_net.net_cv(X, Y, lr_list, epoch_list)
    dr_net.net_roll_test(X, Y, n_roll=4)

    # Predict-then-optimize system
    po_net = bm.pred_then_opt(n_x, n_y, n_obs, gamma=init_params[0], prisk=prisk).double()
    po_net.net_roll_test(X, Y, n_roll=4)

    # Base E2E neural net
    base_net = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                        init_params=init_params, opt_layer='base_mod', perf_loss=perf_loss, 
                        perf_period=perf_period, pred_loss_factor=pred_loss_factor).double()
    base_net.net_cv(X, Y, lr_list, epoch_list)
    base_net.net_roll_test(X, Y, n_roll=4)

    # Equal weight portfolio
    ew_net = bm.equal_weight(n_x, n_y, n_obs)
    ew_net.net_roll_test(X, Y, n_roll=4)

    # Nominal E2E system with fixed gamma
    nom_net_const_gam = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                        init_params=init_params, opt_layer='nominal', perf_loss=perf_loss,
                        perf_period=perf_period, pred_loss_factor=pred_loss_factor, train_gamma=False).double()
    nom_net_const_gam.net_cv(X, Y, lr_list, epoch_list)
    nom_net_const_gam.net_roll_test(X, Y, n_roll=4)

    # DR E2E system with fixed delta
    dr_net_const_del = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                        init_params=init_params, opt_layer='hellinger', perf_loss=perf_loss, 
                        perf_period=perf_period, pred_loss_factor=pred_loss_factor, 
                        train_delta=False).double()
    dr_net_const_del.net_cv(X, Y, lr_list, epoch_list)
    dr_net_const_del.net_roll_test(X, Y, n_roll=4)

    if save_results:
        with open(cache_path+'cached_models.pkl', 'wb') as outp:
            pickle.dump(nom_net, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dr_net, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(po_net, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(base_net, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(ew_net, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(nom_net_const_gam, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dr_net_const_del, outp, pickle.HIGHEST_PROTOCOL)

####################################################################################################
# %% Plots
####################################################################################################

#---------------------------------------------------------------------------------------------------
# %% NUmerical results
#---------------------------------------------------------------------------------------------------
# Validation results table
validation_table = pd.concat((base_net.cv_results.round(3), nom_net.cv_results.val_loss.round(3), 
                            dr_net.cv_results.val_loss.round(3)), axis=1)
validation_table.set_axis(['eta', 'Epochs', 'Base', 'Nom.', 'DR'], axis=1, inplace=True) 

# Out-of-sample summary statistics table
portfolios = ["ew_net", "po_net", "base_net", "nom_net", "dr_net"]
rets =[]
vols = []
SRs = []
for portfolio in portfolios:
    ret = (eval(portfolio).portfolio.rets.tri[-1] ** 
            (1/eval(portfolio).portfolio.rets.tri.shape[0]))**52 - 1
    vol = eval(portfolio).portfolio.vol * np.sqrt(52)
    SR = ret / vol
    rets.append(round(ret*100, ndigits=1))
    vols.append(round(vol*100, ndigits=1))
    SRs.append(round(SR, ndigits=2))

fin_table = pd.DataFrame(np.array([rets, vols, SRs]), columns=['EW', 'PO', 'Base', 'Nom.', 'DR'])

# Wealth evolution plot
portfolio_names = ["Equal Weight", "Pred.-then-Opt.", "Base E2E", "Nom. E2E", "DR E2E"]
portfolio_list = [ew_net.portfolio, po_net.portfolio, base_net.portfolio, nom_net.portfolio, 
                    dr_net.portfolio]
portfolio_colors = ["dimgray", "goldenrod", "forestgreen", "dodgerblue", "salmon"]
pf.wealth_plot(portfolio_list, portfolio_names, portfolio_colors, path="./plots/wealth.pdf")

#---------------------------------------------------------------------------------------------------
# %% Appendix: numerical results
#---------------------------------------------------------------------------------------------------
# Appendix: Validation results table
val_table_app = pd.concat((nom_net_const_gam.cv_results.round(3), 
                            dr_net_const_del.cv_results.val_loss.round(3)), axis=1)
val_table_app.set_axis(['eta', 'Epochs', 'Nom_const_gam', 'DR_const_del'], axis=1, inplace=True) 

# Appendix: Out-of-sample summary statistics table
portfolios = ["nom_net", "nom_net_const_gam", "dr_net", "dr_net_const_del"]
rets =[]
vols = []
SRs = []
for portfolio in portfolios:
    ret = (eval(portfolio).portfolio.rets.tri[-1] ** 
            (1/eval(portfolio).portfolio.rets.tri.shape[0]))**52 - 1
    vol = eval(portfolio).portfolio.vol * np.sqrt(52)
    SR = ret / vol
    rets.append(round(ret*100, ndigits=1))
    vols.append(round(vol*100, ndigits=1))
    SRs.append(round(SR, ndigits=2))

fin_table_app = pd.DataFrame(np.array([rets, vols, SRs]), columns=['cons_gamma', 'Nom.', 'cons_delta', 'DR'])
fin_table_app.set_axis(['Return (%)', 'Volatility (%)', 'Sharpe ratio'], axis=0, inplace=True) 

# Appendix: Wealth evolution plot
portfolio_names = ["Nom. E2E", "Nom. E2E (const. gamma)", "DR E2E", "DR E2E (const. delta)"]
portfolio_list = [nom_net.portfolio, nom_net_const_gam.portfolio, dr_net.portfolio, 
                    dr_net_const_del.portfolio]
portfolio_colors = ["dodgerblue", "mediumblue", "salmon", "firebrick"]
pf.wealth_plot(portfolio_list, portfolio_names, portfolio_colors, 
                path="./plots/wealth_app.pdf")

