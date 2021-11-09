# E2E DRO Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
# Last revision: 08-Nov-2021
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
from e2edro.portfolio import portfolio

####################################################################################################
# DRO CvxpyLayer: Optimizationo problems based on different distance functions
####################################################################################################
#---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
#---------------------------------------------------------------------------------------------------
def tv(n_x, n_y, n_obs, prisk):
    """DRO layer using the 'Total Variation' distance to define the probability ambiguity set.
    From Ben-Tal et al. (2013).
    
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
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al. (2013).

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(obj_aux) - gamma * y_hat @ z
    """

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
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux]
        constraints += [prisk(z, c_aux, ep[i]) - eta_aux <= lambda_aux]

    # Objective function
    objective = cp.Minimize(eta_aux + delta*lambda_aux + (1/n_obs)*cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, delta, gamma], variables=[z])

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
def hellinger(n_x, n_y, n_obs, prisk):
    """DRO layer using the Hellinger distance to define the probability ambiguity set.
    from Ben-Tal et al. (2013).
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    eta_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    const_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable SOC constraint.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al. (2013).

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(obj_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable(n_y, nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    eta_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    const_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    delta = cp.Parameter(nonneg=True)
    gamma = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1,
                mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [const_aux[i] >= 0.5 * (obj_aux[i] - lambda_aux + prisk(z,c_aux,ep[i]) 
                        - eta_aux)]
        constraints += [0.5 * (obj_aux[i] + lambda_aux - prisk(z,c_aux,ep[i]) + eta_aux) >=
                        cp.norm(cp.vstack([lambda_aux, const_aux[i]]))]
        constraints += [prisk(z, c_aux, ep[i]) - eta_aux <= lambda_aux]

    # Objective function
    objective = cp.Minimize(eta_aux + lambda_aux*(delta-1) + (1/n_obs)*cp.sum(obj_aux) - 
                            gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, delta, gamma], variables=[z])

####################################################################################################
# DRO neural network module
####################################################################################################
class e2edro(nn.Module):
    """End-to-end learning NN module

    This module implements a linear prediction layer and a convex quadratic decision layer. The
    module takes the inputs and passes them through the prediction layer. The covariance matrix of
    the corresponding residuals is then calculated. Finally, the residual covariance matrix and
    predictions are passed to the optimization layer to find the optimal decision Z_star.

    Inputs
    n_x: number of features, x_t, in the prediction model
    n_y: number of outputs, y_hat_t, from the prediction model
    n_obs: Number of observations (scenarios) in the complete dataset

    Outputs
    Z_star: (n_obs x n_y) matrix of optimal decisions per scenario
    y_hat: (n_obs x n_y) matrix of predictions
    """

    def __init__(self, n_x, n_y, n_obs, prisk, dro_layer):
        """Layers in the E2E module. 'pred_layer' is a linear regression model. 'z_opt_layer' is
        the optimization layer of the decision variable z and is based on a tractable reformulation
        of the DRO model from Ben-Tal et al. (2013). 'delta' and 'gamma' are registered as 
        nn.Parameters so that they can be 'learned' by the model. 
        """
        super(e2edro, self).__init__()
        
        # Register 'delta' (ambiguity sizing parameter) to make it differentiable
        self.delta = nn.Parameter(torch.rand(1)/5)

        # Register 'gamma' (risk-return trade-off parameter) to make it differentiable
        self.gamma = nn.Parameter(torch.rand(1)/5)

        # LAYER: Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # LAYER: Optimization
        self.opt_layer = dro_layer(n_x, n_y, n_obs, prisk)
        
    def forward(self, X, Y):
        """Forward pass of the NN module

        Inputs
        X: Features. (n_obs x n_x) matrix of timeseries data
        Y: Realizations. (n_obs x n_y) matrix of realizations

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions
        z_t: Optimal solution from 'opt_layer' corresponding to residuals ep and scenario y_hat_t

        Outputs
        Y_hat: Predictions. (n_obs x n_y) matrix of outputs of the prediction layer
        Z_star: Optimal solution. (n_obs x n_y) matrix of optimal decisions. Individual z_t
        solutions are stacked into Z_star.
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {'eps': 1e-12, 'acceleration_lookback': 0, 'max_iters':10000}

        # Optimize z_t per scenario, aggregate solutions into Z_star
        Z_star = []
        for y_hat_t in Y_hat:
            z_t, = self.opt_layer(ep, y_hat_t, self.delta, self.gamma, solver_args=solver_args)
            Z_star.append(z_t)
        Z_star = torch.stack(Z_star).squeeze()

        return Z_star, Y_hat

    def net_train(self, X, Y, Y_train, epochs, perf_loss, pred_loss=torch.nn.MSELoss()):
        """Neural net training module

        Inputs
        X: Features. (n_obs x n_x) matrix of timeseries data
        Y: Realizations. (n_obs x n_y) matrix of realizations
        Y_train: Data for performance loss evaluation. ([n_obs+time_step] x n_y) matrix of 
        realizations. 'time_step' is the number of additional scenarios used during performance 
        evaluation (e.g., to calculate the realized Sharpe ratio)
        epochs: number of training passes
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss: Prediction loss function from fit between predictions Y_hat and realizations Y

        Output
        Trained neural net 'self'
        """
        # Define the optimizer and its parameters
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

        # Train the neural network
        for epoch in range(epochs):
        
            # Input X, predict Y_hat, and optimize to maximize the conditional expectation
            Z_star, Y_hat = self(X, Y)     

            # Loss function: Combination of out-of-sample preformance and prediction
            loss = perf_loss(Z_star, Y_train) + pred_loss(Y_hat, Y)

            # Backpropagation: Clear previous gradients, compute new gradients, update
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            # Ensure that delta, gamma > 0 during backpropagation
            for name, param in self.named_parameters():
                if name=='delta':
                    param.data.clamp_(0.0001)
                if name=='gamma':
                    param.data.clamp_(0.0001)

            # Print loss after every iteration
            print(loss.data.numpy())

            # Print values of delta and gamma to observe their evolution
            for name, param in self.named_parameters():
                if name=='delta':
                    print(name, param.data)
                if name=='gamma':
                    print(name, param.data)

    def net_test(n_obs, X, Y):
        """Neural net testing module

        Use the trained neural net to predict and optimize a running portfolio over the testing 
        dataset. Each portfolio z_t requires that we have residuals (y_hat - y) from t-n_obs to t-1.

        Inputs
        n_obs: Number of residual scenarios to be used during optimization
        X: Feature data. ([n_obs+n_test] x n_x) matrix of timeseries data
        Y: Realizations. ([n_obs+n_test] x n_y) matrix of realizations

        Outputs
        p_opt: portfolio-object containing running portfolio weights, returns, and cumulative
        return
        """
        n_y = Y.shape[1]
        p_opt = portfolio(n_obs, n_y)

        


        return p_opt
