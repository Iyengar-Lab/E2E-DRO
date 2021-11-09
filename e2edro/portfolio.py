# E2E DRO Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
# Last revision: 08-Nov-2021
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np

####################################################################################################
# DRO neural network module
####################################################################################################
class portfolio:

    def __init__(self, n_obs, n_y):

        self.weights = np.zeros((n_obs, n_y))
        self.rets = np.zeros(n_obs)
        self.tri = np.zeros(n_obs)