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
def z_opt(n_y, n_obs):
    """Convex optimization layer in the decision variables 'z'
    
    Variables and parameters
    z: Variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    S: Parameter. (n_obs x n_y) matrix of centered residuals divided by sqrt(n_obs)
    c: Parameter. (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
    
    Constraints
    Budget constraint: sum(z) = 1
    Long-only constraint: z >= 0
    
    Objective
    Minimize_z (1/2) z' * S' * S * z - c' * z
    """
    # Variables
    z = cp.Variable(n_y)

    # Parameters
    S_p = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)

    # Constraints
    constraints = [z >= 0, 
                sum(z)==1]

    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(S_p @ z) - y_hat.T @ z)

    # Optimization problem and layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[S_p, y_hat], variables=[z])

def p_opt(n_obs):
    """Convex optimization layer for projected gradient ascent in the adversarial probabilities 'p'

    The projection takes the form of an optimization problem. We take a step in the gradient direction of p and project it to to the nearest feasible 'p' within the ambiguity set.
    
    Variables and parameters
    p: Variable. (n_obs x 1) vector of adversarial probabilities
    pi: Parameter. (n_obs x n_y) matrix of centered residuals dividedd by sqrt(n_obs)
    y_hat: Parameter. (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)

    Constraints
    From the axioms of probability: p >= 0 and sum(p) == 1
    Distance limit between p and the nominal distribution q, phi(p,q) <= d

    Objective: Minimize_p || p' - p||_2^2 
    Where p' = p_t + gamma * ((1/2) * pi_sq - torch.outer(pi,pi) @ p_t)
    """
    # Nominal (equally-weighted) probability distribution
    q = torch.ones(n_obs) / n_obs

    # Variables
    p = cp.Variable(n_obs)
    u = cp.Variable(n_obs)

    # Parameters
    p_prime = cp.Parameter(n_obs)
    d = cp.Parameter()

    # Constraints
    constraints = [p >= 0, 
                sum(p)==1,
                u >= p - q,
                u >= q - p,
                sum(u) <= d]
                # cp.kl_div(p,q) <= d]

    # Objective
    objective = cp.Minimize(cp.sum_squares(p - p_prime))

    # Optimization problem and layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[p_prime, d], variables=[p])

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
        """
        super(E2E_DRO_Module, self).__init__()
        # Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # # Projected gradient ascent in p
        # self.p_opt_layer = p_opt(n_obs)

        # # Optimization in z
        # self.z_opt_layer = z_opt(n_y, n_obs)

        self.d = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        # Nominal (equally-weighted) probability distribution
        q = torch.ones(n_obs) / n_obs

        # Variables
        p = cp.Variable(n_obs, nonneg=True)
        u = cp.Variable(n_obs, nonneg=True)

        # Parameters
        p_prime = cp.Parameter(n_obs, nonneg=True)
        d = cp.Parameter(1, nonneg=True)

        # Constraints
        constraints = [sum(p)==1,
                    u >= p - q,
                    u >= q - p,
                    sum(u) <= d]
                    # cp.kl_div(p,q) <= d]
        
        # Objective
        objective = cp.Minimize(cp.sum_squares(p - p_prime) )

        # Optimization problem and layer
        problem = cp.Problem(objective, constraints)

        self.p_opt_layer = CvxpyLayer(problem, parameters=[p_prime, d], variables=[p])



        # Variables
        z = cp.Variable(n_y, nonneg=True)

        # Parameters
        S_p = cp.Parameter((n_obs, n_y))
        y_hat = cp.Parameter(n_y)

        # Constraints
        constraints = [sum(z)==1]

        # Objective
        objective = cp.Minimize(10 * cp.sum_squares(S_p @ z) - y_hat.T @ z)

        # Optimization problem and layer
        problem = cp.Problem(objective, constraints)

        self.z_opt_layer = CvxpyLayer(problem, parameters=[S_p, y_hat], variables=[z])

        # Register the robust distance parameter 'd' so that it can be optimized
        # self.register_parameter(name='d', param=nn.Parameter(torch.tensor([0.1])))

    def p_weight_params(self, ep, p):
        """Centering (de-meaning) of residuals and division by sqrt(n_obs). To be used in
        conjuction with cp.sum_squares() to calculate the covariance matrix of the residuals.

        Input
        ep: (n_obs x n_y) matrix of residuals
        p: (n_obs x 1) vector of probabilities (discrete PMF)

        Output
        mu: (n_y x 1) vector of p-weighted average of residuals
        S_sqrt: (n_obs x n_y) matrix, where S_sqrt.T @ S_sqrt = p-weighted covariance
        matrix.
        """
        mu = ep.T @ p
        S_sqrt = ((ep - mu).T * p.sqrt()).T

        return mu, S_sqrt

    def z_weight_params(self, ep, z):
        """z-weighted portfolio residuals

        Input
        ep: (n_obs x n_y) matrix of residuals
        z: (n_y x 1) vector of portfolio weights

        Output
        pi: (n_obs x 1) vector of z-weighted portfolio residuals
        pi_sq: (n_obs x 1) vector of element-wise squared z-weighted portfolio residuals
        """
        pi = ep @ z

        return pi, pi**2
        
    def forward(self, X, Y, P, Z_star):
        """Forward pass of the NN module. 
        X: Features. (n_obs x n_x) matrix of timeseries data
        Y: Realizations. (n_obs x n_y) matrix of realized values.
        Y_hat: Predictions. (n_obs x n_y) matrix of outputs of the prediction layer
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions
        ep_bar: Centered residuals. (n_obs x n_y) matrix of centered residuals divided by sqrt
        (n_obs)
        Z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Each row corresponds
        to a single scenario Y_hat_t, i.e., we ran the optimizer 'n_obs' times to find a 'z_t'
        solution per Y_hat_t. z_t solutions are stacked into Z_star.
        """

        solver_args = {
            # 'mode': 'lsqr',
            # 'solve_method': 'ECOS',
            'verbose': False,
            # 'max_iters': 1000,
            'eps': 1e-10,
            'acceleration_lookback': 0,
            # 'use_indirect': False,
            # 'gpu': False,
            # 'n_jobs_forward': -1,
            # 'n_jobs_backward': -1
            }

        # Number of scenarios
        T = Y.shape[0]

        # Learning rate for gradient ascent in p
        gamma = 0.05

        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(member) for member in X])

        # Calculate residuals
        ep = Y - Y_hat
        Z_new, P_new = [], []
        for t in range(T):
            # Derive parameters from decision z_star and perform projected gradient ascent step in p
            pi, pi_sq = self.z_weight_params(ep, Z_star[t])
            p_prime = P[t] + gamma * ((1/2) * pi_sq - torch.outer(pi,pi) @ P[t].squeeze())
            d = self.d
            d = d.detach()
            p_t, = self.p_opt_layer(p_prime, d, solver_args=solver_args)
            P_new.append(p_t.detach())

            # Derive parameters from distribution p and optimize in z
            mu, S_sqrt = self.p_weight_params(ep, p_t.squeeze())

            z_t, = self.z_opt_layer(S_sqrt, Y_hat[t].squeeze(), solver_args=solver_args)
            Z_new.append(z_t.detach())

        P_new = torch.stack(P_new).squeeze()
        Z_new = torch.stack(Z_new)
        
        return Z_new, P_new, Y_hat

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

P = torch.ones((T,T)) / T
Z_star = torch.ones((T,n)) / n
# d = torch.tensor([0.15], requires_grad=True)

# Train the neural network
for t in range(5):

    # d.grad = None

    # Input X, predict Y_hat, and optimize to maximize the conditional expectation
    Z_star, P, Y_hat = e2enet(X, Y, P, Z_star)   

    # Loss function: Combination of out-of-sample preformance and prediction
    loss = perf_loss(Z_star, Y_test) + pred_loss(Y_hat, Y)

    # Backpropagation: Clear previous gradients, compute new gradients, update
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    # d.data = d.data - 0.1 * d.grad.data
    # d.data[d.data < 0] = 0.0
    
    # Print loss after every iteration
    print(loss.data.numpy())



for name, param in e2enet.named_parameters():
    if param.requires_grad:
        print(name, param.grad)

def KL(a, b):
    
    return torch.sum(a * torch.log(a / b) - a + b, 0)