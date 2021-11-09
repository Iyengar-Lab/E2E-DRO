# End-to-End Distributionally Robust Optimization
#
# Prepared by:    Giorgio Costa (gc2958@columbia.edu)
# Last revision:  31-Oct-2021
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
import pandas as pd
import pandas_datareader as pdr



# import sys
# sys.path.append("/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/E2E-DRO/examples/")
# from e2edro import e2edro_mod as e2e

####################################################################################################
# Define Neural network module
####################################################################################################
def pvar(z, c, x):
    return (x @ z - c)**2

def pmad(z, c, x):
    return cp.abs(x @ z - c)

def psortino(z, c, x):
    return cp.maximum(x @ z - c, 0)**2



class e2edro_nn(nn.Module):
    """End-to-end learning NN module

    This module implements a linear prediction layer and a convex quadratic decision layer. The
    module takes the inputs and passes them through the prediction layer. The covariance matrix of
    the corresponding residuals is then calculated. Finally, the residual covariance matrix and
    predictions are passed to the optimization layer to find the optimal decision Z_star.

    Inputs
    n_x: number of features, x_t, in the prediction model
    n_obs: number of outputs, y_hat_t, from the prediction model
    n_obs: Number of observations (scenarios) in the complete dataset

    Outputs
    Z_star: (n_obs x n_y) matrix of optimal decisions per scenario
    y_hat: (n_obs x n_y) matrix of predictions
    """

    def __init__(self, n_x, n_y, n_obs, prisk):
        """Layers in the E2E module. 'pred_layer' is a linear regression model. 'z_opt_layer' is
        the optimization layer of the decision variable z and is based on a tractable reformulation
        of the DRO model from Ben-Tal et al. (2013). The probability ambiguity set is based on the
        Total Variation distance measure between the adversarial distribution p and the nominal
        distribution q.
        
        The z_opt_layer layer has the following components.

        Variables
        z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
        c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
        lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
        eta_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
        obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.

        Parameters
        ep: (n_obs x n_y) matrix of residuals 
        y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
        returns)
        delta: Scalar. Maximum distance between p and q.

        constraints
        Total budget is equal to 100%, sum(z) == 1
        Long-only positions (no short sales), z >= 0
        All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al. (2013).

        Objective
        Minimize eta_aux + delta*lambda_aux + (1/n_obs)*sum(obj_aux) - y_hat @ z
        """
        super(e2edro_nn, self).__init__()
        # Register 'delta' (ambiguity sizing parameter) to make it differentiable
        self.delta = nn.Parameter(torch.rand(1)/5)

        # Register 'gamma' (risk-return trade-off parameter) to make it differentiable
        self.gamma = nn.Parameter(torch.rand(1)/5)

        # LAYER: Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # LAYER: Optimization
        # Variables
        z = cp.Variable(n_y, nonneg=True)
        c_aux = cp.Variable()
        lambda_aux = cp.Variable(nonneg=True)
        eta_aux = cp.Variable()
        obj_aux = cp.Variable(n_obs)
        mu_aux = cp.Variable()

        # Parameters
        ep = cp.Parameter((n_obs, n_y))
        y_hat = cp.Parameter(n_y)
        delta = cp.Parameter(nonneg=True)
        gamma = cp.Parameter(nonneg=True)
        
        # Constraints
        constraints = [cp.sum(z) == 1,
                    obj_aux >= -lambda_aux,
                    mu_aux == y_hat @ z]
        for i in range(n_obs):
            constraints += [obj_aux[i] >= -lambda_aux]
            # constraints += [obj_aux[i] >= (ep[i] @ z - c_aux)**2 - eta_aux]
            # constraints += [(ep[i] @ z - c_aux)**2 - eta_aux <= lambda_aux]
            constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux]
            constraints += [prisk(z, c_aux, ep[i]) - eta_aux <= lambda_aux]

        # Objective function
        objective = cp.Minimize(eta_aux + delta*lambda_aux + (1/n_obs)*cp.sum(obj_aux) - gamma * mu_aux)

        # Construct optimization problem and differentiable layer
        problem = cp.Problem(objective, constraints)
        self.z_opt_layer = CvxpyLayer(problem, parameters=[ep, y_hat, delta, gamma], variables=[z])
        
    def forward(self, X, Y):
        """Forward pass of the NN module. 
        X: Features. (n_obs x n_x) matrix of timeseries data
        Y: Realizations. (n_obs x n_y) matrix of realized values.
        Y_hat: Predictions. (n_obs x n_y) matrix of outputs of the prediction layer
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions
        Z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row corresponds
        to a single scenario Y_hat_t, i.e., we ran the optimizer 'n_obs' times to find a 'z_t'
        solution per Y_hat_t. z_t solutions are stacked into Z_star.
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {'eps': 1e-12, 'acceleration_lookback': 0, 'max_iters':10000}

        # if self.delta < 0:


        # Optimize z_t per scenario, aggregate solutions into Z_star
        Z_star = []
        for y_hat_t in Y_hat:
            z_t, = self.z_opt_layer(ep, y_hat_t, self.delta, self.gamma, solver_args=solver_args)
            Z_star.append(z_t)
        Z_star = torch.stack(Z_star).squeeze()

        return Z_star, Y_hat

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

####################################################################################################
# EXAMPLE: E2E DRO 
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Generate synthetic data
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
# Load real data from Kenneth French's data library 
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
#---------------------------------------------------------------------------------------------------
asset_df = pdr.get_data_famafrench('10_Industry_Portfolios_daily')[0]
factor_df = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily')[0]
rf_df = factor_df['RF']
factor_df = factor_df.drop(['RF'], axis=1)
mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor_daily')[0]
st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor_daily')[0]
lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor_daily')[0]
factor_df = pd.concat([factor_df, mom_df, st_df, lt_df], axis=1)

# Convert the pd.dataframes to torch.variables to use in the NN module
# Subset the data to a manageable size (e.g., 10 assets, 250 observations)
X_full = Variable(torch.tensor(factor_df.values / 100, dtype=torch.double))[0:300,:]
Y_full = Variable(torch.tensor(asset_df.values / 100, dtype=torch.double))[0:300,0:10]
T = round(X_full.shape[0] * 0.7)
X = X_full[0:T]
Y = Y_full[0:T]
m, n = X.shape[1], Y.shape[1]

#---------------------------------------------------------------------------------------------------
# Train neural net
#---------------------------------------------------------------------------------------------------
# Initialize the neural net
e2enet = e2edro_nn(n_x=m, n_y=n, n_obs=T, prisk=pvar).double()
# e2enet = e2enet.double()  

# Define the optimizer and its parameters
# optimizer = torch.optim.SGD(e2enet.parameters(), lr=0.05, momentum=0.9)
optimizer = torch.optim.Adam(e2enet.parameters(), lr=0.05)

# Prediction loss function
pred_loss = torch.nn.MSELoss()  

# Out-of-sample performance loss function
perf_loss = single_period_over_var_loss

# Train the neural network
for t in range(20):
  
    # Input X, predict Y_hat, and optimize to maximize the conditional expectation
    Z_star, Y_hat = e2enet(X, Y)     

    # Loss function: Combination of out-of-sample preformance and prediction
    loss = perf_loss(Z_star, Y_full) + pred_loss(Y_hat, Y)

    # Backpropagation: Clear previous gradients, compute new gradients, update
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # Ensure that delta, gamma > 0 during backpropagation
    for name, param in e2enet.named_parameters():
        if name=='delta':
            param.data.clamp_(0.0001)
        if name=='gamma':
            param.data.clamp_(0.0001)

    # Print loss after every iteration
    print(loss.data.numpy())

    for name, param in e2enet.named_parameters():
        if name=='delta':
            print(name, param.data)
        if name=='gamma':
            print(name, param.data)



for name, param in e2enet.named_parameters():
    print(name, param.grad.data)

for name, param in e2enet.named_parameters():
    print(name, param.data)

