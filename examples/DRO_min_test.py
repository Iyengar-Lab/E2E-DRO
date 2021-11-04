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

####################################################################################################
# Define Neural network module
####################################################################################################
class E2E_DRO_Module(nn.Module):
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

    def __init__(self, n_x, n_y, n_obs):
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
        rho: Scalar. Maximum distance between p and q.

        constraints
        Total budget is equal to 100%, sum(z) == 1
        Long-only positions (no short sales), z >= 0
        All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al. (2013).

        Objective
        Minimize ta_aux + rho*lambda_aux + (1/n_obs)*sum(obj_aux) - y_hat.T @ z
        """
        super(E2E_DRO_Module, self).__init__()
        # LAYER: Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # LAYER: Optimization
        # Variables
        z = cp.Variable(n_y, nonneg=True)
        c_aux = cp.Variable()
        lambda_aux = cp.Variable(1, nonneg=True)
        eta_aux = cp.Variable()
        obj_aux = cp.Variable(n_obs)

        # Parameters
        ep = cp.Parameter((n_obs, n_y))
        y_hat = cp.Parameter(n_y)
        rho = cp.Parameter(1, nonneg=True)

        # Constraints
        constraints = [sum(z)==1,
                    obj_aux >= -lambda_aux]
        for i in range(n_obs):
            constraints += [obj_aux[i] >= -lambda_aux]
            constraints += [obj_aux[i] >= (ep[i].T @ z - c_aux)**2 - eta_aux]
            constraints += [(ep[i].T @ z - c_aux)**2 - eta_aux <= lambda_aux]

        # Objective function
        objective = cp.Minimize(eta_aux + rho*lambda_aux + (1/n_obs)*sum(obj_aux) - y_hat.T @ z)

        # Construct optimization problem and differentiable layer
        problem = cp.Problem(objective, constraints)
        self.z_opt_layer = CvxpyLayer(problem, parameters=[ep, y_hat, rho], variables=[z])
        
    def forward(self, X, Y, rho):
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
        Y_hat = torch.stack([self.pred_layer(member) for member in X])

        # Calculate residuals and process them
        ep = Y - Y_hat

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {'eps': 1e-10, 'acceleration_lookback': 0}

        # Optimize z_t per scenario, aggregate solutions into Z_star
        Z_star = []
        for member in Y_hat:
            z_t, = self.z_opt_layer(ep, member, rho, solver_args=solver_args)
            Z_star.append(z_t.detach())
        Z_star = torch.stack(Z_star)

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

####################################################################################################
# EXAMPLE: E2E DRO 
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Generate synthetic data
#---------------------------------------------------------------------------------------------------
torch.manual_seed(1)
# Number of observations T, features m and outputs n
T, m, n = 112, 3, 5

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
# Train neural net
#---------------------------------------------------------------------------------------------------
# Define the neural net
e2enet = E2E_DRO_Module(n_x=m, n_y=n, n_obs=T)     

# Define the optimizer and its parameters
optimizer = torch.optim.Adam(e2enet.parameters(), lr=0.1)

# Prediction loss function
pred_loss = torch.nn.MSELoss()  

# Out-of-sample performance loss function
perf_loss = sharpe_loss

# TO-DO: Make the ambiguity set sizing parameter 'rho' learnable by the neural net
rho = torch.tensor([0.1])

# Train the neural network
for t in range(5):
  
    # Input X, predict Y_hat, and optimize to maximize the conditional expectation
    Z_star, Y_hat = e2enet(X, Y, rho)     

    # Loss function: Combination of out-of-sample preformance and prediction
    loss = perf_loss(Z_star, Y_test) + pred_loss(Y_hat, Y)

    # Backpropagation: Clear previous gradients, compute new gradients, update
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # Print loss after every iteration
    print(loss.data.numpy())

# Check the optimal parameters from the neural net
for name, param in e2enet.named_parameters():
    if param.requires_grad:
        print(name, param.grad)


