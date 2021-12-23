# Naive Model Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

my_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL"
import sys
sys.path.append(my_path+"/E2E-DRO")

import e2edro.RiskFunctions as rf
import e2edro.PortfolioClasses as pc
import e2edro.e2edro2 as e2e

from importlib import reload 
reload(pc)

model_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/saved_models/"

####################################################################################################
# Naive 'predict-then-optimize'
####################################################################################################
class pred_then_opt(nn.Module):
    """Naive 'predict-then-optimize' portfolio construction module
    """
    def __init__(self, n_x, n_y, n_obs, gamma=0.1, prisk='p_var'):
        """Naive 'predict-then-optimize' portfolio construction module

        This NN module implements a linear prediction layer 'pred_layer' and an optimization layer 
        'opt_layer'. The model is 'naive' since it optimizes each layer separately. 

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        
        Output
        pred_then_opt: nn.Module object 
        """
        super(pred_then_opt, self).__init__()

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Store 'gamma' (risk-return trade-off parameter)
        self.gamma = torch.tensor(gamma, dtype=torch.double)

        # LAYER: OLS linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)
        self.pred_layer.weight.requires_grad = False
        self.pred_layer.bias.requires_grad = False
        
        # LAYER: Optimization
        self.opt_layer = e2e.nominal(n_y, n_obs, eval('rf.'+prisk))

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the predict-then-optimize module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        y_hat = Y_hat[-1]

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {'solve_method': 'ECOS'}

        # Optimize z per scenario
        # Determine whether nominal or dro model
        z_star, = self.opt_layer(ep, y_hat, self.gamma, solver_args=solver_args)

        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=5):
        """Neural net rolling window out-of-sample test
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            X_train, Y_train = X.train(), Y.train()
            X_train.insert(0,'ones', 1.0)

            X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
            Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
        
            Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
            Theta = Theta.T
            del X_train, Y_train

            with torch.no_grad():
                self.pred_layer.bias.copy_(Theta[:,0])
                self.pred_layer.weight.copy_(Theta[:,1:])

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]

                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio