""" Lab 2 """

import numpy as np
import matplotlib.pyplot as plt

from biopack import integrate, DEFAULT, parse_args
import biolog

from ex3_pendulum import PendulumParameters, pendulum_system


DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]


def pendulum_integration(state, time=None, parameters=None):
    """ Function for system integration """
    return pendulum_system(state[0], state[1], time, parameters)[:, 0]


def exercise3():
    """ Exercise 3 """
    parameters = PendulumParameters()  # Checkout pendulum.py for more info
    biolog.info(parameters)
    # Simulation parameters
    time = np.arange(0, 30, 0.01)  # Simulation time
    x0 = [0.1, 0.0]  # Initial state

    # To use/modify pendulum parameters (See PendulumParameters documentation):
    # parameters.g = 9.81  # Gravity constant
    # parameters.L = 1.  # Length
    # parameters.d = 0.3  # damping
    # parameters.sin = np.sin  # Sine function
    # parameters.dry = False  # Use dry friction (True or False)

    # Example of system integration (Similar to lab1)
    # (NOTE: pendulum_equation must be imlpemented first)
    biolog.debug("Running integration example")
    res = integrate(pendulum_integration, x0, time, args=(parameters,))
    res.plot_state("State")
    res.plot_phase("Phase")

    # Evolutions
    # Write code here (You can add functions for the different cases)
    biolog.warning(
        "Evolution of pendulum in normal conditions must be implemented"
    )
    biolog.warning(
        "Evolution of pendulum without damping must be implemented"
    )
    biolog.warning(
        "Evolution of pendulum with perturbations must be implemented"
    )
    biolog.warning(
        "Evolution of pendulum with dry friction must be implemented"
    )

    # Show plots of all results
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    parse_args()
    exercise3()

