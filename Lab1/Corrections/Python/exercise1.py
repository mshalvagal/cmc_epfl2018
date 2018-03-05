""" Lab 1 - Exercise 1 """

from ex1_functions import function, function_rk, analytic_function
from ex1_integration import (
    euler_integrate,
    ode_integrate,
    ode_integrate_rk,
    plot_integration_methods
)
from ex1_errors import compute_error

import numpy as np
import matplotlib.pyplot as plt

import biolog
from biopack import Result, DEFAULT


def error(method_state, analytical_state, method):
    """ Compute error of an ode method based on analytical solution """
    err = np.sum(np.abs(analytical_state - method_state))/len(method_state)
    err_max = max(np.abs(analytical_state - method_state))
    biolog.debug(
        "Obtained an error of {} using {} method (max={}, len={})".format(
            err, method, err_max, len(method_state)
        )
    )
    return err


def exercise1():
    """ Exercise 1 """
    # Setup
    biolog.info("Running exercise 1")

    # Setup
    time_max = 5  # Maximum simulation time
    time_step = 0.2  # Time step for ODE integration in simulation
    x0 = np.array([1.])  # Initial state

    # Integration methods (Exercises 1.a - 1.d)
    biolog.info("Running function integration using different methods")


    # Analytical (1.a)
    time = np.arange(0, time_max, time_step)  # Time vector
    x_a = analytic_function(time)
    analytical = Result(x_a, time) if x_a is not None else None

    # Euler (1.b)
    euler = euler_integrate(function, x0, time_max, time_step)
    error(euler.state, analytical.state, "Euler")

    # ODE (1.c)
    ode = ode_integrate(function, x0, time_max, time_step)
    error(ode.state, analytical.state, "LSODA")

    # ODE Runge-Kutta (1.c)
    ode_rk = ode_integrate_rk(function_rk, x0, time_max, time_step)
    error(ode_rk.state, analytical.state, "Runge-Kutta")

    # Euler with lower time step (1.d)
    euler_time_step = 0.05
    time_small = np.arange(0, time_max, euler_time_step)  # Time vector
    x_a = analytic_function(time_small)
    analytical_ts_small = Result(x_a, time_small)
    euler_ts_small = (
        euler_integrate(function, x0, time_max, euler_time_step)
        if euler_time_step is not None
        else None
    )
    error(
        euler_ts_small.state,
        analytical_ts_small.state,
        "Euler (smaller ts)"
    )

    # Plot integration results
    plot_integration_methods(
        analytical=analytical, euler=euler,
        ode=ode, ode_rk=ode_rk, euler_ts_small=euler_ts_small,
        euler_timestep=time_step, euler_timestep_small=euler_time_step
    )

    # Error analysis (Exercise 1.e)
    biolog.info("Studying integration error using different methods")
    dt_list = np.logspace(-3, 0, 20)  # List of timesteps (powers of 10)
    integration_errors = [["L1", 1], ["L2", 2], ["Linf", 0]]
    methods = [
        ["Euler", euler_integrate, function],
        ["Lsoda", ode_integrate, function],
        ["RK", ode_integrate_rk, function_rk]
    ]
    for error_name, error_index in integration_errors:
        for name, integration_function, f in methods:
            compute_error(
                f, analytic_function, integration_function, x0, dt_list,
                time_max=time_max,
                figure=error_name,
                label=name,
                args=(error_index,)
            )

    # Show plots of all results
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == "__main__":
    from biopack import parse_args
    parse_args()
    exercise1()