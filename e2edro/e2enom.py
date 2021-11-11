# E2E Nominal Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
# Last revision: 10-Nov-2021
#
####################################################################################################
## Import libraries
####################################################################################################
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

####################################################################################################
# SlidingWindow torch Dataset to index data to use a sliding window
####################################################################################################
class SlidingWindow(Dataset):
    """Sliding window dataset constructor
    """
    def __init__(self, X, Y, n_obs, perf_period):
        """Construct a sliding (i.e., rolling) window dataset from a complete timeseries dataset

        Inputs
        X: Complete feature dataset
        Y: Complete realizations dataset
        n_obs: Number of scenarios in the window
        perf_period: Number of scenarios in the 'performance window' used to evaluate out-of-sample
        performance. The 'performance window' is also a sliding window

        Output
        Dataset where each element is the tuple (x, y, y_perf)
        x: Feature window (dim: [n_obs+1] x n_x)
        y: Realizations window (dim: n_obs x n_y)
        y_perf: Window of forward-looking (i.e., future) realizations (dim: perf_period x n_y)

        Note: For each feature window 'x', the last scenario x_t is reserved for prediction and
        optimization. Therefore, no pair in 'y' is required (it is assumed the pair y_T is not yet
        observable)
        """
        self.X = X
        self.Y = Y
        self.window = n_obs+1
        self.perf_period = perf_period

    def __getitem__(self, index):
        x = self.X[index:index+self.window]
        y = self.Y[index:index+self.window-1]
        y_perf = self.Y[index+self.window-1:index+self.window+self.perf_period]
        return (x, y, y_perf)

    def __len__(self):
        return len(self.X) - self.window - self.perf_period

####################################################################################################
# Portfolio-object to store out-of-sample results
####################################################################################################
class portfolio:
    """Portfolio object
    """
    def __init__(self, len_test, n_y):
        """Portfolio object. Stores the NN out-of-sample results

        Inputs
        len_test: Number of scenarios in the out-of-sample evaluation period
        n_y: Number of assets in the portfolio

        Output
        Portfolio object with fields:
        weights: Asset weights per period (dim: len_test x n_y)
        rets: Realized portfolio returns (dim: len_test x 1)
        tri: Total return index (i.e., absolute cumulative return) (dim: len_test x 1)
        mean: Average return over the out-of-sample evaluation period (dim: scalar)
        vol: Volatility (i.e., standard deviation of the returns) (dim: scalar)
        sharpe: pseudo-Sharpe ratio defined as 'mean / vol' (dim: scalar)
        """
        self.weights = np.zeros((len_test, n_y))
        self.rets = np.zeros(len_test)

    def stats(self):
        self.tri = np.cumprod(self.rets + 1)
        self.mean = (self.tri[-1])**(1/len(self.tri)) - 1
        self.vol = np.std(self.rets)
        self.sharpe = self.mean / self.vol

####################################################################################################
# opt_layer: CvxpyLayer that declares the portfolio optimization problem
####################################################################################################
def opt_layer(n_y, n_obs, prisk):
    """Nominal optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)

    Objective
    Minimize (1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux
    """
    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i])]

    # Objective function
    objective = cp.Minimize((1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma], variables=[z])

####################################################################################################
# Nominal neural network module
####################################################################################################
class e2e(nn.Module):
    """End-to-end nominal learning neural net module
    """
    def __init__(self, n_x, n_y, n_obs, prisk):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a convex optimization
        layer 'opt_layer'. 'gamma' is declared as a nn.Parameter so that it can be 'learned'.

        Inputs
        n_x: number of inputs (i.e., features) in the prediction model
        n_y: number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: Portfolio risk function. Used in the opt_layer

        Output
        e2e: nn.Module object 
        """
        super(e2e, self).__init__()

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs
        
        # Register 'gamma' (risk-return trade-off parameter) to make it differentiable
        self.gamma = nn.Parameter(torch.rand(1)/5)

        # LAYER: Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # LAYER: Optimization
        self.opt_layer = opt_layer(n_y, n_obs, prisk)
        
    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the NN module

        Inputs
        X: Features. ([n_obs+1] x n_x) matrix of timeseries data
        Y: Realizations. (n_obs x n_y) matrix of realizations

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_t: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        y_hat = Y_hat[-1]

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {'eps': 1e-10, 'acceleration_lookback': 0, 'max_iters':15000}

        # Optimize z_t per scenario, aggregate solutions into Z_star
        z_star, = self.opt_layer(ep, y_hat, self.gamma, solver_args=solver_args)

        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, X, Y, epochs, perf_loss, pred_loss=torch.nn.MSELoss(), perf_period=22):
        """Neural net training module

        Inputs
        X: Features. (T x n_x) tensor of timeseries data
        Y: Realizations. (T x n_y) tensor of realizations
        epochs: number of training passes
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss: Prediction loss function from fit between predictions Y_hat and realizations Y
        perf_period: Number of lookahead realizations used in 'perf_loss()'

        Output
        Trained neural net 'self'
        """
        train_loader = DataLoader(SlidingWindow(X, Y, self.n_obs, perf_period))

        # Define the optimizer and its parameters
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

        # Train the neural network
        for epoch in range(epochs):
            optimizer.zero_grad() 
            for n, (x, y, y_perf) in enumerate(train_loader):
                z_star, y_hat = self(x.squeeze(), y.squeeze())
                if pred_loss is None:
                    loss = perf_loss(z_star, y_perf.squeeze())
                else:
                    loss = (perf_loss(z_star, y_perf.squeeze()) + 
                            pred_loss(y_hat, y_perf.squeeze()[0]))
                loss.backward()
            optimizer.step()

            # Ensure that gamma > 0 during backpropagation. Print their values to observe
            # their evolution
            for name, param in self.named_parameters():
                if name=='gamma':
                    print(name, param.data)
                    param.data.clamp_(0.0001)

            # Print loss after every iteration
            print(loss.data.numpy())

    #-----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_test(self, X, Y):
        """Neural net testing module

        Use the trained neural net to predict and optimize a running portfolio over the testing 
        dataset. Each portfolio z_t requires that we have residuals (y_hat - y) from t-n_obs to t-1.

        Inputs
        n_obs: Number of residual scenarios to be used during optimization
        X: Feature data. ([n_obs+n_test] x n_x) matrix of timeseries data
        Y: Realizations. ([n_obs+n_test] x n_y) matrix of realizations

        Output
        p_opt: object containing running portfolio weights, returns, and cumulative returns
        """
        # Prepare the data for testing as a SlidingWindow dataset
        test_loader = DataLoader(SlidingWindow(X, Y, self.n_obs, 0))

        # Declare portfolio object to hold the test results
        p_opt = portfolio(len(test_loader), self.n_y)

        with torch.no_grad():
            for t, (x, y, y_perf) in enumerate(test_loader):
                # Predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Store portfolio weights and returns for each time step 't'
                p_opt.weights[t] = z_star.squeeze()
                p_opt.rets[t] = y_perf.squeeze() @ p_opt.weights[t]

        # Calculate the portfolio statistics using the realized portfolio returns
        p_opt.stats()

        return p_opt
