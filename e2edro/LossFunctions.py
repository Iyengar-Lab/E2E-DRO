# Financial performance loss functions for E2E learning framework
#
# Prepared by:    Giorgio Costa (gc2958@columbia.edu)
# Last revision:  08-nov-2021
#
####################################################################################################
## Import libraries
####################################################################################################
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

####################################################################################################
# Performance loss functions
####################################################################################################
def sharpe_loss(Z_star, Y):
    """Loss function based on the out-of-sample Sharpe ratio

    Compute the out-of-sample Sharpe ratio of the portfolio z_t over the next 12 time steps. The
    loss is aggregated for all z_t in Z_star and averaged over the number of observations. We use a
    simplified version of the Sharpe ratio, SR = realized mean / realized std dev.

    Inputs
    Z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row of Z_star is z_t
    for t = 1, ..., T. 
    Y: Realizations. (n_obs x n_y) matrix of realized values.

    Output
    Aggregate loss for all t = 1, ..., T, divided by n_obs
    """
    loss = 0
    i = -1
    time_step = 12
    for z_t in Z_star:
        i += 1
        Y_t = Y[i:time_step+i]
        loss += -torch.mean(Y_t @ z_t) / torch.std(Y_t @ z_t)
    return loss / i

def single_period_loss(Z_star, Y):
    """Loss function based on the out-of-sample portfolio return

    Compute the out-of-sample portfolio return for portfolio z_t over the next time step. The
    loss is aggregated for all z_t in Z_star and averaged over the number of observations.

    Inputs
    Z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row of Z_star is z_t
    for t = 1, ..., T. 
    Y: Realizations. (n_obs x n_y) matrix of realized values.

    Output
    Aggregate loss for all t = 1, ..., T, divided by n_obs
    """
    loss = 0
    i = -1
    for z_t in Z_star:
        i += 1
        loss += -Y[i] @ z_t
    return loss / i

def single_period_over_var_loss(Z_star, Y):
    """Loss function based on the out-of-sample portfolio return

    Compute the out-of-sample portfolio return for portfolio z_t over the next time step. The
    loss is aggregated for all z_t in Z_star and averaged over the number of observations.

    Inputs
    Z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row of Z_star is z_t
    for t = 1, ..., T. 
    Y: Realizations. (n_obs x n_y) matrix of realized values.

    Output
    Aggregate loss for all t = 1, ..., T, divided by n_obs
    """
    loss = 0
    i = -1
    time_step = 22
    for z_t in Z_star:
        i += 1
        Y_t = Y[i:time_step+i]
        loss += -Y[i] @ z_t / torch.std(Y_t @ z_t)
    return loss / i