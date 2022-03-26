# Risk functions module
# 
# This module defines the finanical risk measures to be used in the optimization layer of the E2E
# problem.
#
####################################################################################################
## Import libraries
####################################################################################################
import cvxpy as cp

####################################################################################################
# Define risk functions
####################################################################################################
def p_var(z, c, x):
    """Variance
    Inputs
    z: (n x 1) vector of portfolio weights (decision variable)
    c: Scalar. Centering parameter that serves as a proxy to the expected value (auxiliary variable)
    x: (n x 1) vector of realized returns (data)

    Output: Single squared deviation. 
    Note: This function is only one component of the portfolio variance, and must be aggregated 
    over all scenarios 'x' to recover the complete variance
    """
    return cp.square(x @ z - c)

def p_mad(z, c, x):
    """Mean absolute deviation (MAD)
    Inputs
    z: (n x 1) vector of portfolio weights (decision variable)
    c: Scalar. Centering parameter that serves as a proxy to the expected value (auxiliary variable)
    x: (n x 1) vector of realized returns (data)

    Output: Single absolute deviation. 
    Note: This function is only one component of the portfolio MAD, and must be aggregated 
    over all scenarios 'x' to recover the complete MAD
    """
    return cp.abs(x @ z - c)