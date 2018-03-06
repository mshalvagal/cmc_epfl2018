""" System functions """

import numpy as np


def analytic_function(t):
    """ Analytic funtion x(t) (To be written)

    Notes:
    - To write exponential(x) in Python with numpy: np.exp(x)
    Example:
    >>> np.exp(0.0)
    1.0

    This function should simply return the state with same shape as input t
    (i.e. as a list or an areray)
    """
    # COMPLETE CODE
    
    return 5-4*np.exp(-2*t)

def function(x, *_):
    """ Exercise 1 ODE equation

    Second argument is only required for use with odeint format and can be
    ignored

    Output expected:
    >>> function(0.)
    10.0
    >>> function(5.)
    0.0

    This function should return the derivative x_dot with same shape as x
    (i.e. as a list or an areray)
    """
    # COMPLETE CODE
    dxdt = 2*(5-x)
    if dxdt is None:
        raise NotImplementedError("ODE has not been implemented")
    return dxdt


def function_rk(t=None, x=None):
    """ Exercise 1 ODE equation for Runge-Kutta """
    return function(x, t)