# Distributionally Robust End-to-End Portfolio Construction

This repository contains the source code to reproduce the numerical results in our paper [Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134). Our paper introduces a robust end-to-end (E2E) learning system where the final decision layer is based on a distributionally robust (DR) optimization model for portfolio construction. This repository was created by [Giorgio Costa](https://gcosta151.github.io) and [Garud Iyengar](http://www.columbia.edu/~gi10/), and belongs to Iyengar Lab in the IEOR Department at Columbia University.

# Introduction
We propose an end-to-end distributionally robust system for portfolio construction that integrates the asset return prediction model with a distributionally robust portfolio optimization model. We also show how to learn the risk-tolerance parameter and the degree of robustness directly from data. End-to-end systems have an advantage in that information can be communicated between the prediction and decision layers during training, allowing the parameters to be trained for the final task rather than solely for predictive performance. However, existing end-to-end systems are not able to quantify and correct for the impact of model risk on the decision layer. Our proposed distributionally robust end-to-end portfolio selection system explicitly accounts for the impact of model risk. The decision layer chooses portfolios by solving a minimax problem where the distribution of the asset returns is assumed to belong to an ambiguity set centered around a nominal distribution. Using convex duality, we recast the minimax problem in a form that allows for efficient training of the end-to-end system.

# Dependencies
- Python 3.x/numpy/scipy/pandas/matplotlib
- cvxpy 1.x
- cvxpylayers 0.1.x
- PyTorch 1.x
- pandas_datareader/alpha_vantage

# Usage
This repository contains all the files and data used to generate the experimental results in our paper. To reproduce the experimental results, please refer to the main.py file.

Anyone wishing to design their own robust portfolios should refer to the e2edro module. The e2edro module contains the following files:
- e2edro.py: Allows one to construct DR E2E objects. 
- RiskFunctions.py: Includes two alternative deviation risk functions that can be called by e2edro objects. These are the portfolio variance and portfolio mean absolute deviation.
- LossFunctions.py: Includes three alternative loss functions for end-to-end learning. These are the portfolio return, the portfolio Sharpe ratio, and the portfolio single-period return over the portfolio multi-period standard deviation.
- PortfolioClasses.py: Includes multiple objects used to store experimental data and results, as well as portfolio backtest results. 
- BaseModels.py: Allows one to construct naive models without end-to-end learning. These are used as competing models in the numerical experiments.
- DataLoad.py: Provides code to download factor data from Kenneth French's data library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and asset data from AlphaVantage (www.alphavantage.co). It also includes functtions to generate synthetic data. 

# Licensing
Unless otherwise stated, the source code is copyright of Columbia University and licensed under the Apache 2.0 License.
