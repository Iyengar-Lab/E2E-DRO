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
    predictions are passed to the optimization layer to find the optimal decision z_star.

    Inputs
    n_x: number of features, x_t, in the prediction model
    n_obs: number of outputs, y_hat_t, from the prediction model
    n_obs: Number of observations (scenarios) in the complete dataset

    Outputs
    z_star: (n_obs x n_y) matrix of optimal decisions per scenario
    y_hat: (n_obs x n_y) matrix of predictions
    """

    def __init__(self, n_x, n_y, n_obs):
        """Layers in the E2E module. 'pred_layer' is a linear regression model. 'z_opt_layer' is 
        the convex quadratic optimization layer of the decision variable z. 'p_opt_layer' is the concave quadratic optimization layer of the adversarial probability variable p. 
        
        The z_opt_layer layer has the following components:
        z: Variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
        S: Parameter. (n_obs x n_y) matrix of centered residuals dividedd by sqrt(n_obs)
        c: Parameter. (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
        Cconstraint: Total budget is equal to 100%, sum(z) == 1
        Cconstraint: Long-only positions (no short sales), z >= 0
        Objective: Minimize_z (1/2) z' * S' * S * z - c' * z
        """
        super(E2E_DRO_Module, self).__init__()
        # LAYER: Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # LAYER: Optimization
        # Variables
        z = cp.Variable(n_y)
        c_aux = cp.Variable()
        lambda_aux = cp.Variable()
        eta_aux = cp.Variable()
        pi_aux = cp.Variable(n_obs)

        # Parameters
        ep = cp.Parameter((n_obs, n_y))
        y_hat = cp.Parameter(n_y)
        rho = cp.Parameter(1, nonneg=True)

        # Constraints
        constraints = [z >= 0, 
                    sum(z)==1,
                    lambda_aux >= 0]
        obj_expr = 0
        for i in range(n_obs):
            constraints += [(ep[i].T @ z - c_aux)**2 <= lambda_aux + eta_aux]
            # obj_expr += self.cvx_conjugate( ((ep[i].T @ z - c_aux)**2 - eta_aux) / lambda_aux )
            obj_expr +=  self.cvx_conjugate((ep[i].T @ z - c_aux)**2 - eta_aux)

            # constraints += [pi_aux[i] == ep[i].T @ z - c_aux]
            # constraints += [pi_aux[i]**2 <= lambda_aux + eta_aux]
            # obj_expr += self.cvx_conjugate( (pi_aux[i]**2 - eta_aux) / lambda_aux )

        # objective = cp.Minimize(eta_aux + rho*lambda_aux + (lambda_aux/n_obs)*obj_expr - y_hat.T @ z)
        objective = cp.Minimize(eta_aux + rho*lambda_aux + (1/n_obs)*obj_expr - y_hat.T @ z)

        problem = cp.Problem(objective, constraints)
        self.z_opt_layer = CvxpyLayer(problem, parameters=[ep, y_hat, rho], variables=[z])

    def cvx_conjugate(self, s):
        # return s / (1-s)
        return s

    def p_weighted(self, ep, p):
        """Centering (de-meaning) of residuals and division by sqrt(n_obs). To be used in
        conjuction with cp.sum_squares() to calculate the covariance matrix of the residuals.

        Input
        ep: (n_obs x n_y) matrix of residuals
        p: (n_obs x 1) vector of probabilities (discrete PMF)

        Output
        mu: (n_y x 1) vector of p-weighted average of residuals
        Sigma_sqrt: (n_y x n_obs) matrix, where Sigma_sqrt.T @ Sigma_sqrt = p-weighted covariance
        matrix.
        """
        mu = ep.T @ p
        ep -= mu
        Sigma_sqrt = p.sqrt().diag() @ ep.T

        return mu, Sigma_sqrt
        
    def forward(self, X, Y):
        """Forward pass of the NN module. 
        X: Features. (n_obs x n_x) matrix of timeseries data
        Y: Realizations. (n_obs x n_y) matrix of realized values.
        Y_hat: Predictions. (n_obs x n_y) matrix of outputs of the prediction layer
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions
        ep_bar: Centered residuals. (n_obs x n_y) matrix of centered residuals divided by sqrt
        (n_obs)
        z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row corresponds
        to a single scenario Y_hat_t, i.e., we ran the optimizer 'n_obs' times to find a 'z_t'
        solution per Y_hat_t. z_t solutions are stacked into z_star.
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(member) for member in X])

        # Calculate residuals and process them
        ep = Y - Y_hat

        # Optimize z_t per scenario, aggregate solutions into z_star
        z_star = []
        rho = 0.1
        for member in Y_hat:
            z_t, = self.z_opt_layer(ep, member, rho)
            z_star.append(z_t)
        z_star = torch.stack(z_star)

        return z_star, Y_hat

def sharpe_loss(z_star, Y):
    """Loss function based on the out-of-sample Sharpe ratio

    Compute the out-of-sample Sharpe ratio of the portfolio z_t over the next 12 time steps. The
    loss is aggregated for all z_t in z_star and averaged over the number of observations. We use a
    simplified version of the Sharpe ratio, SR = realized mean / realized std dev.

    Inputs
    z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row of z_star is z_t
    for t = 1, ..., T. 
    Y: Realizations. (n_obs x n_y) matrix of realized values.

    Output
    Aggregate loss for all t = 1, ..., T, divided by n_obs
    """
    loss = 0
    i = -1
    time_step = 12
    for z_t in z_star:
        i += 1
        Y_t = Y[i:time_step+i]
        loss += -torch.mean(Y_t @ z_t) / torch.std(Y_t @ z_t)
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
e2enet = E2E_DRO_Module(n_x=m, n_y=n, n_obs=T)     

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




