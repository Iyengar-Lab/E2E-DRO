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
class TableModule(nn.Module):
    """MapTable module: 
    https://github.com/amdegroot/pytorch-containers/blob/master/README.md#maptable

    MapTable is a container for a single module which will be applied to all input elements. The
    member module is cloned as necessary to process all input elements.
    """
    def __init__(self, n_in, n_out):
        super(TableModule, self).__init__()
        self.layer = nn.Linear(n_in, n_out)
        
    def forward(self, x):
        y_hat = torch.stack([self.layer(member) for member in x])
        return y_hat

class E2EModule(nn.Module):
    """End-to-end learning NN module

    This module implements a linear prediction layer and a convex quadratic decision layer. The
    module takes the inputs and passes them through the prediction layer. The covariance matrix of
    the corresponding residuals is then calculated. Finally, the residual covariance matrix and
    predictions are passed to the optimization layer to find the optimal decision z_star.

    Inputs:
    n_x: number of features, x_t, in the prediction model
    n_obs: number of outputs, y_hat_t, from the prediction model
    n_obs: Number of observations (scenarios) in the complete dataset

    Output:
    z_star: (n_obs x n_y) matrix of optimal decisions per scenario
    y_hat: (n_obs x n_y) matrix of predictions
    """

    def __init__(self, n_x, n_y, n_obs):
        """Layers in the E2E module. 'pred_layer' is a linear regression model. 'opt_layer' is a 
        convex quadratic optimization layer based on the CVXPY and CvxpyLayer libraries. The
        optimization layer has the following components:

        Variables and parameters
        z: Variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
        S: Parameter. (n_obs x n_y) matrix of centered residuals dividedd by sqrt(n_obs)
        c: Parameter. (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
        
        Constraints: 
        Budget constraint: sum(z) = 1
        Long-only constraint: z >= 0
        
        Problem:
        Minimize_z (1/2) z' * S' * S * z - c' * z
        """
        super(E2EModule, self).__init__()
        # Linear prediction layer
        self.pred_layer = nn.Linear(n_x, n_y)

        # Optimization layer
        z = cp.Variable(n_y)
        S = cp.Parameter((n_obs, n_y))
        c = cp.Parameter(n_y)
        constraints = [z >= 0, sum(z)==1]
        objective = cp.Minimize(0.5 * cp.sum_squares(S @ z) - c.T @ z)
        problem = cp.Problem(objective, constraints)
        self.opt_layer = CvxpyLayer(problem, parameters=[S, c], variables=[z])

    def cov(self, x):
        """Centering (de-meaning) of residuals and division by sqrt(n_obs). To be used in
        conjuction with cp.sum_squares() to calculate the covariance matrix of the residuals
        """
        sqrtT = torch.sqrt(torch.as_tensor(x.shape[-1]))
        mean = torch.mean(x, dim=-1).unsqueeze(-1)
        x -= mean
        return 1/sqrtT * x.T
        
    def forward(self, x, y):
        """Forward pass of the NN module. 
        x: Inputs. (n_obs x n_x) matrix of feature data
        y: Realizations. (n_obs x n_y) matrix of realized values.
        y_hat: Predictions. (n_obs x n_y) matrix of outputs of the prediction layer
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions
        ep_bar: Centered residuals. (n_obs x n_y) matrix of centered residuals divided by sqrt
        (n_obs)
        z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row corresponds
        to a single scenario y_hat_t, i.e., we ran the optimizer 'n_obs' times to find a 'z_prime'
        solution per y_hat_t. z_prime solutions are stacked into z_star.
        """
        # Predict y_hat from x
        y_hat = torch.stack([self.pred_layer(member) for member in x])

        # Calculate residuals and process them
        ep = y - y_hat
        ep_bar = self.cov(ep.T)

        # Optimize z_prime per scenario, aggregate solutions into z_star
        z_star = []
        for member in y_hat:
            z_prime, = self.opt_layer(ep_bar, member)
            z_star.append(z_prime)
        z_star = torch.stack(z_star)

        return z_star, y_hat

def sharpe_loss(z_star, Y):
    loss = 0
    i = -1
    time_step = 12
    for z in z_star:
        i += 1
        Y_t = Y[i:time_step+i]
        loss += -torch.mean(Y_t @ z) / torch.std(Y_t @ z)
    return loss / i

####################################################################################################
# EXAMPLE: Nominal E2E learning
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
e2enet = E2EModule(n_x=m, n_y=n, n_obs=T)     

# Define the optimizer and its parameters
optimizer = torch.optim.Adam(e2enet.parameters(), lr=0.1)

# Prediction loss function
pred_loss = torch.nn.MSELoss()  

# Out-of-sample performance loss function
perf_loss = sharpe_loss

# Train the neural network
for t in range(200):
  
    # Input X, predict Y_hat, and optimize to maximize the conditional expectation
    z_star, Y_hat = e2enet(X, Y)     

    # Loss function: Combination of out-of-sample preformance and prediction
    loss = perf_loss(z_star, Y_test) + pred_loss(Y_hat, Y)

    # Backpropagation: Clear previous gradients, compute new gradients, update
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # Print loss after every iteration
    print(loss.data.numpy())




