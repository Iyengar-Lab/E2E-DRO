# Distributionally Robust End-to-End Portfolio Construction

This repository presents an end-to-end (E2E) learning system where the final decision layer is based on a distributionally robust (DR) optimization model for portfolio construction. This repository was created by [Giorgio Costa](https://gcosta151.github.io) and [Garud Iyengar](http://www.columbia.edu/~gi10/), and belongs to Iyengar Lab in the IEOR Department at Columbia University.

# Introduction
We propose an end-to-end distributionally robust system for portfolio construction that integrates the asset return prediction model with a portfolio optimization model. Furthermore, we also show how to learn the risk-tolerance parameter and the degree of robustness directly from data. End-to-end systems have an advantage in that information can be communicated between the prediction and decision layers during training, allowing the parameters to be trained for the final task rather than solely for predictive performance. However, existing end-to-end systems are not able to quantify and correct for the impact of model risk on the decision layer. We propose a distributionally robust end-to-end portfolio selection system that explicitly accounts for the impact of model risk. The decision layer chooses portfolios by solving a minimax problem where the distribution of the asset returns is assumed to belong to an ambiguity set centered around a nominal distribution. Using convex duality, we recast the minimax problem in a form that allows for efficient training of the end-to-end system.

# ICML & Journal
ICML: This folder accompanies our ICML 2022 paper on DR E2E Portfolio Construction. The folder contains all the files that were originally submitted with our paper and can be used to reproduce the experimental results. 

Journal: This folder accompanies the journal version of this paper. The folder contains all the files used to generate the experimental results. The Journal code base includes significant improvements and additional detail compared to the ICML code base. Therefore, we recommend using the Journal code base over of the ICML version. 

# Usage
To reproduce the ICML experimental results, please refer to the main.py file within the ICML folder. 

To reproduce the Journal experimental results, please refer to the main.py file within the Journal folder.

Anyone wishing to design their own robust portfolios should refer to the e2edro module within the Journal folder. The e2edro module contains the following files:
- e2edro.py: Allows one to construct DR E2E objects. 
- RiskFunctions.py: Includes two alternative deviation risk functions that can be called by e2edro objects. These are the portfolio variance and portfolio mean absolute deviation.
- LossFunctions.py: Includes three alternative loss functions for end-to-end learning. These are the portfolio return, the portfolio Sharpe ratio, and the portfolio single-period return over the portfolio multi-period standard deviation.
- PortfolioClasses.py: Includes multiple objects used to store experimental data and results, as well as portfolio backtest results. 
- BaseModels.py: Allows one to construct naive models without end-to-end learning. These are used as competing models in the numerical experiments.
- DataLoad.py: Provides code to download factor data from Kenneth French's data library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and asset data from AlphaVantage (www.alphavantage.co). It also includes functtions to generate synthetic data. 

