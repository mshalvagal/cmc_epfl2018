""" Lab 1 - Exercise 1 """

from ex1_functions import function, function_rk, analytic_function
from ex1_integration import (
    example_integrate,
    euler_integrate,
    ode_integrate,
    ode_integrate_rk,
    plot_integration_methods
)
from ex1_errors import compute_error, plot_error

import numpy as np
import matplotlib.pyplot as plt

import biolog
from biopack import Result, DEFAULT




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

    # Example
    biolog.debug("Running example plot for integration (remove)")
    example = example_integrate(x0, time_max, time_step)
    example.plot_state(figure="Example", label="Example", marker=".")

    # Analytical (1.a)
    time = np.arange(0, time_max, time_step)  # Time vector
    x_a = analytic_function(time)
    analytical = Result(x_a, time) if x_a is not None else None

    # Euler (1.b)
    euler = euler_integrate(function, x0, time_max, time_step)

    # ODE (1.c)
    ode = ode_integrate(function, x0, time_max, time_step)

    # ODE Runge-Kutta (1.c)
    ode_rk = ode_integrate_rk(function_rk, x0, time_max, time_step)

    # Euler with lower time step (1.d)
    euler_time_step = 0.2/4
    euler_ts_small = (
        euler_integrate(function, x0, time_max, euler_time_step)
        if euler_time_step is not None
        else None
    )

    # Plot integration results
    plot_integration_methods(
        analytical=analytical, euler=euler,
        ode=ode, ode_rk=ode_rk, euler_ts_small=euler_ts_small,
        euler_timestep=time_step, euler_timestep_small=euler_time_step
    )

    # Error analysis (Exercise 1.e)
    dt_list = np.logspace(-4,0)
    compute_error(function, analytic_function, euler_integrate, x0, dt_list, label="Euler")
    compute_error(function, analytic_function, ode_integrate, x0, dt_list, label="ODE")
    

    # Show plots of all results
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == "__main__":
    from biopack import parse_args
    parse_args()
    exercise1()