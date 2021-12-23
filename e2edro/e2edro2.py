# E2E DRO Module
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

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl

model_path = "/Users/giorgio/Library/Mobile Documents/com~apple~CloudDocs/Documents/Google Drive/Research Projects/2021/E2E DRL/saved_models/"

from importlib import reload 
reload(pc)

####################################################################################################
# CvxpyLayers: Differentiable optimization layers (nominal and distributionally robust)
####################################################################################################
#---------------------------------------------------------------------------------------------------
# nominal: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def nominal(n_y, n_obs, prisk):
    """Nominal optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
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

#---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
#---------------------------------------------------------------------------------------------------
def tv(n_y, n_obs, prisk):
    """DRO layer using the 'Total Variation' distance to define the probability ambiguity set.
    From Ben-Tal et al. (2013).
    Total Variation: sum_t abs(p_t - q_t) <= delta

    Inputs
    n_y: Number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
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
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                obj_aux >= -lambda_aux,
                mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux]
        constraints += [prisk(z, c_aux, ep[i]) - eta_aux <= lambda_aux]

    # Objective function
    objective = cp.Minimize(eta_aux + delta * lambda_aux + (1/n_obs) * cp.sum(obj_aux)
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
def hellinger(n_y, n_obs, prisk):
    """DRO layer using the Hellinger distance to define the probability ambiguity set.
    from Ben-Tal et al. (2013).
    Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
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
    objective = cp.Minimize(eta_aux + lambda_aux * (delta-1) + (1/n_obs) * cp.sum(obj_aux) 
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)
    
    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

####################################################################################################
# DRO neural network module
####################################################################################################
class e2e(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_loss_factor=0.5, perf_period=12, train_pred=True, set_seed=None):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Determine whether to train the prediction layer (or keep it fixed)
        set_seed: Random seed to use for reproducibility

        Output
        e2e: nn.Module object 
        """
        super(e2e, self).__init__()

        if set_seed is not None:
            torch.manual_seed(set_seed)

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Prediction loss function
        if pred_loss_factor is not None:
            self.pred_loss_factor = pred_loss_factor
            self.pred_loss = torch.nn.MSELoss()
        else:
            self.pred_loss = None

        # Define performance loss
        self.perf_loss = eval('lf.'+perf_loss)

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = nn.Parameter(torch.rand(1)/20 + 0.005)

        # LAYER: Linear prediction (initialize to pre-trained values if requested)
        self.pred_layer = nn.Linear(n_x, n_y)
        self.pred_layer.weight.requires_grad = train_pred
        self.pred_layer.bias.requires_grad = train_pred

        # LAYER: Optimization
        self.opt_layer = eval(opt_layer)(n_y, n_obs, eval('rf.'+prisk))
        
        # Record the model design: nominal or DRO
        if opt_layer == 'nominal':
            self.nominal = True 
            self.model_type = 'nom'
        else:
            self.nominal = False
            self.model_type = 'dro'

            # Register 'delta' (ambiguity sizing parameter) for DRO model
            self.delta = nn.Parameter(torch.rand(1)/15 + 0.025)

        # Store initial model
        torch.save(self.state_dict(), model_path+self.model_type+'_initial_state')
        
    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
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
        solver_args = {'solve_method': 'ECOS'}

        # Optimize z per scenario
        # Determine whether nominal or dro model
        if self.nominal:
            z_star, = self.opt_layer(ep, y_hat, self.gamma, solver_args=solver_args)
        else:
            z_star, = self.opt_layer(ep, y_hat, self.gamma, self.delta,
                        solver_args=solver_args)

        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(epochs):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad() 
            for t, (x, y, y_perf) in enumerate(train_set):

                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                if self.pred_loss is None:
                    loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                else:
                    loss = (1/n_train) * (self.perf_loss(z_star, y_perf.squeeze()) + 
                    (self.pred_loss_factor/self.n_y) * self.pred_loss(y_hat, y_perf.squeeze()[0]))

                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
        
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                if name=='delta':
                    param.data.clamp_(0.0001)

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    if self.pred_loss_factor is None:
                        loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    else:
                        loss = (1/n_val) * (self.perf_loss(z_val, y_perf.squeeze()) + 
                        (self.pred_loss_factor/self.n_y)*self.pred_loss(y_val, y_perf.squeeze()[0]))
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

        # If val_set is None, then save the parameters of the fully trained model
        else:
            if self.nominal:
                torch.save(self.state_dict(), model_path+'nom_net_full')
            else:
                torch.save(self.state_dict(), model_path+'dro_net_full')
            print("Trained model saved")

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net training module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                if self.nominal:
                    print(f"Training E2E nominal model: lr={lr}, epochs={epochs}")
                else:
                    print(f"Training E2E DR model: lr={lr}, epochs={epochs}")

                val_loss_tot = []
                for i in range(n_val):

                    # Partition training dataset into training and validation subset
                    split = [0.2*(i+1), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(model_path+self.model_type+'_initial_state'))

                    # Initialize the prediction layer weights to OLS regression weights
                    X_train, Y_train = X_temp.train(), Y_temp.train()
                    X_train.insert(0,'ones', 1.0)

                    X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                    Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
                
                    Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                    Theta = Theta.T
                    del X_train, Y_train

                    with torch.no_grad():
                        self.pred_layer.bias.copy_(Theta[:,0])
                        self.pred_layer.weight.copy_(Theta[:,1:])

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {i+1} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(model_path+self.model_type+'_results.pkl')

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        if self.nominal:
            print(f"E2E nominal with optimal hyperparameters: lr={self.lr}, epochs={self.epochs}")
        else:
            print(f"CV E2E dro with optimal  hyperparameters: lr={self.lr}, epochs={self.epochs}")

    #-----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=5, lr=None, epochs=None):
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
            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 
                                                    self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(model_path+self.model_type+'_initial_state'))

            # Initialize the prediction layer weights to OLS regression weights
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

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

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

    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]