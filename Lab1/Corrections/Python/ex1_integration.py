""" Exercise 1 - integration """

import numpy as np
from scipy.integrate import odeint, ode

import biolog
from biopack import Result


def example_integrate(x0, time_max, time_step):
    """ Example to show how to use Python

    Note that the Result class takes x and time as input, and is used to
    facilitate the plotting of the results, where x and time are lists or
    arrays

    Additionally, using:

    r = Result(x, t)

    You can then use r.state or r.time to extract the original x and t

    You can then plot the state in function of time with:
    r.plot_state(figure="Name of figure")
    """
    time = np.arange(0, time_max, time_step)
    x = np.zeros([len(time), len(x0)])
    for i, ti in enumerate(time[:-1]):
        x[i+1] = x[i] + 1e-2*ti
    # for i in range(len(time[:-1])):  # Also possible
    #     x[i+1] = x[i] + 0.2
    return Result(x, time)


def euler_integrate(fun, x0, time_max, time_step):
    """ Integrate function using Euler method

    - fun: Function df/dt = f(x, t) to integrate
    - x0: Initial state
    - time_tot: Total time to simulate [s]
    - time_step: Time step [s]

    For loop in Python:

    >>> a = [0, 0, 0]  # Same as [0 for _ in range(3)] (List comprehension)
    >>> for i in range(3):
    ...     a[i] = i
    >>> print(a)
    [0, 1, 2]

    Creating a matrix of zeros in python:

    >>> state = np.zeros([3, 2])
    >>> print(state)
    [[0. 0.]
     [0. 0.]
     [0. 0.]]

    For loop for a state in python

    >>> state = np.zeros([3, 2])
    >>> state[0, 0], state[0, 1] = 1, 2
    >>> for i, time in enumerate([0, 0.1, 0.2]):
    ...     state[i, 0] += time
    ...     state[i, 1] -= time
    >>> print(state)
    [[ 1.   2. ]
     [ 0.1 -0.1]
     [ 0.2 -0.2]]

    Make sure to use the Result class similarly to the example_integrate
    found above (i.e. Result(x, time))
    """
    time = np.arange(0, time_max, time_step)
    x = np.zeros([len(time), len(x0)])
    # COMPLETE CODE
    x[0] = x0
    for i, ti in enumerate(time[:-1]):
        x[i+1] = x[i] + fun(x[i], ti)*time_step
    return Result(x, time)


def ode_integrate(fun, x0, time_max, time_step):
    """ Integrate function using Euler method

    - fun: Function df/dt = f(x, t) to integrate
    - x0: Initial state
    - time_tot: Total time to simulate [s]
    - time_step: Time step [s]

    Use odeint from the scipy library for integration

    Make sure to then use the Result class similarly to the example_integrate
    found above (i.e. Result(x, time))
    """
    time = np.arange(0, time_max, time_step)
    # COMPLETE CODE
    # x = odeint(fun, x0, time)
    # return Result(x, time)
    x, infodict = odeint(fun, x0, time, full_output=True)
    return Result(x, time)


def ode_integrate_rk(fun_rk, x0, time_max, time_step):
    """ Integrate function using Euler method

    - fun: Function df/dt = f(x, t) to integrate
    - x0: Initial state
    - time_tot: Total time to simulate [s]
    - time_step: Time step [s]

    For Runge-Kutta, use:
    solver = ode(fun)
    solver.set_integrator('dopri5')
    solver.set_initial_value(x0, time[0])
    xi = solver.integrate(ti)

    Note that solver.integrate(ti) only integrates for one time step at time ti
    (where ti is the current time), so you will need to use a for loop

    Make sure to use the Result class similarly to the example_integrate
    found above (i.e. Result(x, time))
    """
    time = np.arange(0, time_max, time_step)
    solver = ode(fun_rk)
    solver.set_integrator('dopri5')
    solver.set_initial_value(x0, time[0])
    x = np.array([solver.integrate(t) for t in time])
    return Result(x, time)


def plot_integration_methods(**kwargs):
    """ Plot integration methods results """
    biolog.info("Plotting integration results")

    # Results
    analytical = kwargs.pop("analytical", None)
    euler = kwargs.pop("euler", None)
    ode = kwargs.pop("ode", None)
    ode_rk = kwargs.pop("ode_rk", None)
    euler_ts_small = kwargs.pop("euler_ts_small", None)

    # Figures
    fig_all = kwargs.pop(
        "figure",
        "Integration methods".replace(" ", "_")
    )
    fig_ts = "Integration methods smaller ts".replace(" ", "_")
    fig_euler = "Euler integration".replace(" ", "_")
    d = "."

    # Time steps information
    et_val = kwargs.pop("euler_timestep", None)
    ets_val = kwargs.pop("euler_timestep_small", None)
    et = " (ts={})".format(et_val) if et_val is not None else ""
    ets = " (ts={})".format(ets_val) if ets_val is not None else ""

    # Analytical
    if analytical is not None:
        analytical.plot_state(figure=fig_euler, label="Analytical", marker=d)

    # Euler
    if euler is not None:
        euler.plot_state(figure=fig_euler, label="Euler"+et, marker=d)

    # ODE
    ls = " "
    ode_plot = False
    if ode is not None:
        if ode_plot is False:
            analytical.plot_state(figure=fig_all, label="Analytical", marker=d)
            euler.plot_state(figure=fig_all, label="Euler"+et, marker=d)
            ode_plot = True
        ode.plot_state(figure=fig_all, label="LSODA", linestyle=ls, marker="x")

    # ODE Runge-Kutta
    if ode_rk is not None:
        if ode_plot is False:
            analytical.plot_state(figure=fig_all, label="Analytical", marker=d)
            euler.plot_state(figure=fig_all, label="Euler"+et, marker=d)
        ode_rk.plot_state(figure=fig_all, label="RK", linestyle=ls, marker=d)

    # Euler with lower time step
    if euler_ts_small is not None:
        euler_ts_small.plot_state(
            figure=fig_ts,
            label="Euler"+ets
        )
        analytical.plot_state(figure=fig_ts, label="Analytical", marker=d)

    return