# Naive Model Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import cvxpy as cp
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

my_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL"
import sys
sys.path.append(my_path+"/E2E-DRO")
import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc

from importlib import reload 
reload(pc)

####################################################################################################
# DRO neural network module
####################################################################################################
class pred_then_opt:

    def __init__(self, n_x, n_y, n_obs, prisk='p_var', gamma=1.5):
        """
        Predict-then-optimize model
        
        This is a naive counterpart to the E2E learning framework and serves as a benchmark. 

        Inputs
        n_x: number of inputs (i.e., features) in the prediction model
        n_y: number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: Portfolio risk function. Used in the optimization problem
        gamma: Scalar. Trade-off between conditional expected return and model error.

        Output
        pred_then_opt object 
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs
        self.prisk = eval('rf.'+prisk)
        self.gamma = gamma

    #-----------------------------------------------------------------------------------------------
    # pred: Prediction from OLS regression model
    #-----------------------------------------------------------------------------------------------
    def pred(self, X, Y):
        """Prediction from OLS regression model

        Inputs
        X: Features. ([n_obs+1] x n_x) tensor of timeseries data
        Y: Realizations. (n_obs x n_y) tensor of realizations

        Outputs
        y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
        ep: (n_obs x n_y) matrix of residuals 
        """
        Y_hat = X @ self.Theta
        ep = Y - Y_hat[:-1].squeeze()

        return Y_hat[-1].squeeze(), ep
        

    #-----------------------------------------------------------------------------------------------
    # opt: CVXPY portfolio optimization problem
    #-----------------------------------------------------------------------------------------------
    def opt(self, y_hat, ep):
        """Nominal optimization problem

        Inputs
        ep: (n_obs x n_y) matrix of residuals 
        y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
        
        Variables
        z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
        c_aux: Auxiliary Variable. Scalar
        obj_aux: Auxiliary Variable. (n_obs x 1) vector.
        mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.
        
        Constraints
        Total budget is equal to 100%, sum(z) == 1
        Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)

        Objective
        Minimize (1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux
        """
        # Variables
        z = cp.Variable((self.n_y,1), nonneg=True)
        c_aux = cp.Variable()
        obj_aux = cp.Variable(self.n_obs)
        mu_aux = cp.Variable()
        
        # Constraints
        constraints = [cp.sum(z) == 1,
                    mu_aux == y_hat @ z]
        for i in range(self.n_obs):
            constraints += [obj_aux[i] >= self.prisk(z, c_aux, ep[i])]

        # Objective function
        objective = cp.Minimize((1/self.n_obs) * cp.sum(obj_aux) - self.gamma * mu_aux)

        # Construct optimization problem
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.SCS, eps=1e-10, max_iters=25000)

        return z.value

    #-----------------------------------------------------------------------------------------------
    # train: Naive model training through OLS regression
    #-----------------------------------------------------------------------------------------------
    def train(self, X, Y):
        """Training through OLS regression model

        Inputs
        X: Features. TrainValTest object of feature timeseries data
        Y: Realizations. TrainValTest object of asset time series data

        Outputs
        self.Theta: Weights. (n_y x [n_x+1]) tensor of regression weights (including intercept)
        """
        # Add a column of ones to the feature dataset 
        X_train = X.train()
        X_train.insert(0,'ones',1.0)

        # Subset the train data and convert to torch tensor
        X = Variable(torch.tensor(X_train.values, dtype=torch.double))
        Y = Variable(torch.tensor(Y.train().values, dtype=torch.double))
        
        # Compute OLS weights
        self.Theta = torch.inverse(X.T @ X) @ (X.T @ Y)

    #-----------------------------------------------------------------------------------------------
    # test: Naive model test
    #-----------------------------------------------------------------------------------------------
    def test(self, X, Y):
        """Out-of-sample test of the simple predict-then-optimize model

        Inputs
        X: Features. TrainValTest object of feature timeseries data
        Y: Realizations. TrainValTest object of asset time series data

        Output
        portfolio: object containing running portfolio weights, returns, and cumulative returns
        """
        # Store the test period dates
        dates = Y.test().index

        # Add a column of oones to account for the intercept
        X_test = X.test()
        X_test.insert(0,'ones',1.0)

        # Prepare the data for testing as a SlidingWindow object
        test_loader = DataLoader(pc.SlidingWindow(X_test, Y.test(), self.n_obs, 0))

        # Free memory
        del X_test

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(test_loader), self.n_y, dates)

        for t, (x, y, y_perf) in enumerate(test_loader):
            # Predict and optimize
            y_hat, ep = self.pred(x.squeeze(), y.squeeze())
            z_star = self.opt(y_hat, ep)

            # Store portfolio weights and returns for each time step 't'
            portfolio.weights[t] = z_star.squeeze()
            portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        return portfolio