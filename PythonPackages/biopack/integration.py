""" ODE integration """

from scipy.integrate import odeint
from .results import (
    Result,
    MultipleResultsODE
)


def integrate(ode, x0, time, args=()):
    """ Integrate ode """
    x = odeint(ode, x0, time, args=tuple(args))
    return Result(x, time, ode, args)


def integrate_multiple(ode, x0_list, time, args=()):
    """ Integrate ode for multiple initial states given by x0_list """
    x = [odeint(ode, x0, time, args=tuple(args)) for x0 in x0_list]
    return MultipleResultsODE(x, time, ode, args)
