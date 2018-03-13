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
    res.plot_state("State 1")
    res.plot_phase("Phase 1")

    # Evolutions
    # Write code here (You can add functions for the different cases)
    x0 = [np.pi, 0.0]
    res = integrate(pendulum_integration, x0, time, args=(parameters,))
    res.plot_state("State 2")
    res.plot_phase("Phase 2")

    parameters.d = 0.0
    res = integrate(pendulum_integration, [0.1, 0.0], time, args=(parameters,))
    res.plot_state("State_oscillation")
    res.plot_phase("Phase_oscillation")
    
    temp = np.size(time)/10
    x0 = [0.1, 0.0]
    for i in range(10):
        res = integrate(pendulum_integration, x0, time[temp*i:temp*(i+1)], args=(parameters,))
        x0[1] = np.pi*np.random.random()
        res.plot_state("State_perturbation")
        res.plot_phase("Phase_perturbation")

    parameters.d = 0.3
    parameters.dry = True
    res_dry = integrate(pendulum_integration, [np.pi/2, -0.1], time, args=(parameters,))
    res_dry.plot_state("State_dry")
    res_dry.plot_phase("Phase_dry")

    # Show plots of all results
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    parse_args()
    exercise3()

