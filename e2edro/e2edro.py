# E2E DRO Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc

from importlib import reload 
reload(pc)

model_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/saved_models/"

####################################################################################################
# CvxpyLayers: Differentiable optimization layers (nominal and distributionally robust)
####################################################################################################
#---------------------------------------------------------------------------------------------------
# nominal: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def nominal(n_y, n_obs, prisk, turnover=False):
    """Nominal optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    turnover: Optional. Determine whether the portfolio turnover rate should be constrained
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

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
    if turnover:
        z_t = cp.Parameter((n_y,1), nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i])]
    if turnover:
        constraints += [cp.sum(cp.abs(z - z_t)) <= 1.0]

    # Objective function
    objective = cp.Minimize((1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    if turnover:
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, z_t], variables=[z])
    else:
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma], variables=[z])

#---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
#---------------------------------------------------------------------------------------------------
def tv(n_y, n_obs, prisk, turnover=False):
    """DRO layer using the 'Total Variation' distance to define the probability ambiguity set.
    From Ben-Tal et al. (2013).
    Total Variation: sum_t abs(p_t - q_t) <= delta

    Inputs
    n_y: Number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    turnover: Optional. Determine whether the portfolio turnover rate should be constrained
    
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
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
    (2013).

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(obj_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    eta_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)
    if turnover:
        z_t = cp.Parameter((n_y,1), nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                obj_aux >= -lambda_aux,
                mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux]
        constraints += [prisk(z, c_aux, ep[i]) - eta_aux <= lambda_aux]
    if turnover:
        constraints += [cp.sum(cp.abs(z - z_t)) <= 1.0]

    # Objective function
    objective = cp.Minimize(eta_aux + delta * lambda_aux + (1/n_obs) * cp.sum(obj_aux)
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    if turnover:
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta, z_t], variables=[z])
    else: 
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
def hellinger(n_y, n_obs, prisk, turnover=False):
    """DRO layer using the Hellinger distance to define the probability ambiguity set.
    from Ben-Tal et al. (2013).
    Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    turnover: Optional. Determine whether the portfolio turnover rate should be constrained
    
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
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
    (2013).

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(obj_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    eta_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    const_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)
    if turnover:
        z_t = cp.Parameter((n_y,1), nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1,
                mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [const_aux[i] >= 0.5 * (obj_aux[i] - lambda_aux + prisk(z,c_aux,ep[i]) 
                        - eta_aux)]
        constraints += [0.5 * (obj_aux[i] + lambda_aux - prisk(z,c_aux,ep[i]) + eta_aux) >=
                        cp.norm(cp.vstack([lambda_aux, const_aux[i]]))]
        constraints += [prisk(z, c_aux, ep[i]) - eta_aux <= lambda_aux]
    if turnover:
        constraints += [cp.sum(cp.abs(z - z_t)) <= 1.0]

    # Objective function
    objective = cp.Minimize(eta_aux + lambda_aux * (delta-1) + (1/n_obs) * cp.sum(obj_aux) 
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)
    
    if turnover:
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta, z_t], variables=[z])
    else:
        return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

####################################################################################################
# DRO neural network module
####################################################################################################
class e2e(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, prisk='p_var', opt_layer='nominal', turnover=False, 
                iter_id=None):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: number of inputs (i.e., features) in the prediction model
        n_y: number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: Portfolio risk function. Used in the opt_layer
        dro_layer: CvxpyLayer-object corresponding to a convex DRO probelm

        Output
        e2e: nn.Module object 
        """
        super(e2e, self).__init__()

        if iter_id is not None:
            torch.manual_seed(iter_id+100)

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = nn.Parameter(torch.rand(1)/20 + 0.005)

        # LAYER: Linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)

        # LAYER: Optimization
        self.opt_layer = eval(opt_layer)(n_y, n_obs, eval('rf.'+prisk), turnover)
        
        # Record the model design: with or without turnover rate constraints
        self.turnover = turnover

        # Record the model design: nominal or DRO
        if opt_layer == 'nominal':
            self.nominal = True 
        else:
            self.nominal = False

            # Register 'delta' (ambiguity sizing parameter) for DRO model
            self.delta = nn.Parameter(torch.rand(1)/15 + 0.025)
        
    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y, z_t=None):
        """Forward pass of the NN module

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
        solver_args = {'eps': 1e-12, 'acceleration_lookback': 0, 'max_iters':25000}

        # Optimize z per scenario
        # Determine whether nominal or dro model, and whether turnover is constrained or not
        if self.turnover:
            if self.nominal:
                z_star, = self.opt_layer(ep, y_hat, self.gamma, z_t, solver_args=solver_args)
            else:
                z_star, = self.opt_layer(ep, y_hat, self.gamma, self.delta, z_t,
                            solver_args=solver_args)
        else: 
            if self.nominal:
                z_star, = self.opt_layer(ep, y_hat, self.gamma, solver_args=solver_args)
            else:
                z_star, = self.opt_layer(ep, y_hat, self.gamma, self.delta,
                            solver_args=solver_args)

        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, X, Y, epochs=50, lr=0.025, perf_loss='sharpe_loss', pred_loss_factor=0.5, 
                    perf_period=12, pre_params=None, iter_id=''):
        """Neural net training module

        Inputs
        X: Features. TrainValTest object of feature timeseries data
        Y: Realizations. TrainValTest object of asset time series data
        epochs: number of training passes
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'

        Output
        results.df: results dataframe with running loss, gamma and delta values (dim: epochs x 3)
        """

        # Initialize prediction layer parameters to a pre-trained value
        if pre_params is not None:
            with torch.no_grad():
                self.pred_layer.bias.copy_(pre_params[:,0])
                self.pred_layer.weight.copy_(pre_params[:,1:])

        # Prepare the training data as a SlidingWindow object
        train_loader = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, perf_period))
        n_train = len(train_loader)

        # Prepare the validation data as a SlidingWindow object
        validate = False
        if X.val().shape[0] != 0:
            val_loader = DataLoader(pc.SlidingWindow(X.val(), Y.val(), self.n_obs, perf_period))
            n_val = len(val_loader)
            
            # Initialize value for the "best running model"
            best_tot_val_loss = float("inf")
            validate = True

        # Prediction loss function
        if pred_loss_factor is not None:
            pred_loss = torch.nn.MSELoss()

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Define performance loss
        perf_loss = eval('lf.'+perf_loss)

        # Declare InSample object to hold the training results
        results = pc.InSample()

        # Placeholder for running portfolio
        z_t = None

        # Train the neural network
        if self.nominal:
            print("============== Training E2E nominal model ==============")
        else:
            print("============== Training E2E DRO model ==============")

        for epoch in range(epochs):
        
            # TRAINING: forward + backward pass
            tot_loss = 0
            optimizer.zero_grad() 
            for t, (x, y, y_perf) in enumerate(train_loader):

                # Set an initial portfolio if the turnover rate will be constrained
                if self.turnover and t==0:
                    z_t = torch.ones((self.n_y,1), dtype=torch.float64) / self.n_y

                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze(), z_t)
                if self.turnover:
                    z_t = z_star.detach().clone()

                # Loss function
                if pred_loss_factor is None:
                    loss = (1/n_train) * perf_loss(z_star, y_perf.squeeze())
                else:
                    loss = (1/n_train) * (perf_loss(z_star, y_perf.squeeze()) + 
                           (pred_loss_factor/self.n_y) * pred_loss(y_hat, y_perf.squeeze()[0]))

                # Backward pass: backpropagation
                loss.backward()
                tot_loss += loss.item()
            
            # Update parameters
            optimizer.step()
            results.loss.append(tot_loss)

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                    results.gamma.append(param.data.numpy()[0])
                if name=='delta':
                    param.data.clamp_(0.0001)
                    results.delta.append(param.data.numpy()[0])

            # Optional: VALIDATION
            if validate:
                tot_val_loss = 0
                with torch.no_grad():
                    for t, (x, y, y_perf) in enumerate(val_loader):

                        # Set an initial portfolio if the turnover rate will be constrained
                        if self.turnover and t==0:
                            z_t = torch.ones((self.n_y,1), dtype=torch.float64) / self.n_y

                        # Predict and optimize
                        z_val, y_val = self(x.squeeze(), y.squeeze(), z_t)
                        if self.turnover:
                            z_t = z_val.detach().clone()

                        # Loss function
                        if pred_loss_factor is None:
                            val_loss = (1/n_val) * perf_loss(z_val, y_perf.squeeze())
                        else:
                            val_loss = (1/n_val) * (perf_loss(z_val, y_perf.squeeze()) + 
                               (pred_loss_factor/self.n_y) * pred_loss(y_val, y_perf.squeeze()[0]))
                        tot_val_loss += val_loss.item()
                results.val_loss.append(tot_val_loss)

                # Save running model
                if tot_val_loss < best_tot_val_loss:
                    best_tot_val_loss = tot_val_loss
                    if self.nominal:
                        torch.save(self.state_dict(), model_path+'nom_net_best'+iter_id)
                    else:
                        torch.save(self.state_dict(), model_path+'dro_net_best'+iter_id)
                    print("Model saved")

            # PRINT RESULTS
            if validate:
                if self.nominal:
                    print("Epoch: %d/%d,  " %(epoch+1,epochs),  
                        "TrainLoss: %.3f,  " %tot_loss, 
                        "ValLoss: %.3f,  " %tot_val_loss,
                        "gamma: %.3f" %results.gamma[epoch])
                else:
                    print("Epoch: %d/%d, " %(epoch+1,epochs),  
                        "TrainLoss: %.3f, " %tot_loss, 
                        "ValLoss: %.3f, " %tot_val_loss,
                        "gamma: %.3f, " %results.gamma[epoch],
                        "delta: %.3f" %results.delta[epoch])
            else: 
                if self.nominal:
                    print("Epoch: %d/%d,  " %(epoch+1,epochs),  
                        "TrainLoss: %.3f,  " %tot_loss, 
                        "gamma: %.3f" %results.gamma[epoch])
                else:
                    print("Epoch: %d/%d, " %(epoch+1,epochs),  
                        "TrainLoss: %.3f, " %tot_loss, 
                        "gamma: %.3f, " %results.gamma[epoch],
                        "delta: %.3f" %results.delta[epoch])

        if self.nominal:
            torch.save(self.state_dict(), model_path+'nom_net_final'+iter_id)
        else:
            torch.save(self.state_dict(), model_path+'dro_net_final'+iter_id)
        print("Final model saved")

        return results.df()

    #-----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_test(self, X, Y):
        """Neural net testing module

        Use the trained neural net to predict and optimize a running portfolio over the testing 
        dataset. Each portfolio z_t requires that we have residuals (y_hat - y) from t-n_obs to t-1.

        Inputs
        n_obs: Number of residual scenarios to be used during optimization
        X: Features. TrainValTest object of feature timeseries data
        Y: Realizations. TrainValTest object of asset time series data

        Output
        portfolio: object containing running portfolio weights, returns, and cumulative returns
        """
        dates = Y.test().index

        # Placeholder for running portfolio
        z_t = None

        # Prepare the data for testing as a SlidingWindow object
        test_loader = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(test_loader), self.n_y, dates)

        with torch.no_grad():
            for t, (x, y, y_perf) in enumerate(test_loader):
                
                # Set an initial portfolio if the turnover rate will be constrained
                if self.turnover and t==0:
                    z_t = torch.ones((self.n_y,1), dtype=torch.float64) / self.n_y

                # Predict and optimize
                z_star, _ = self(x.squeeze(), y.squeeze(), z_t)
                if self.turnover:
                    z_t = z_star.clone()

                # Store portfolio weights and returns for each time step 't'
                portfolio.weights[t] = z_star.squeeze()
                portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        return portfolio
