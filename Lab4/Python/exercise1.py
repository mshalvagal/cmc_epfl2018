""" Lab 4 """

import numpy as np
import matplotlib.pyplot as plt
from biopack import integrate, DEFAULT, parse_args
import biolog
from SystemParameters import PendulumParameters
from lab4_pendulum import pendulum_system

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]


def pendulum_integration(state, time, *args, **kwargs):
    """ Function for system integration """
    biolog.warning(
        "Pendulum equation with spring and damper must be implemented")  # l_S
    return pendulum_system(state[0], state[1], *args, **kwargs)[:, 0]


def exercise1():
    """ Exercise 1  """
    biolog.info("Executing Lab 4 : Exercise 1")
    parameters = PendulumParameters()
    biolog.info(
        "Find more information about Pendulum Parameters in SystemParameters.py")
    biolog.warning("Loading default pendulum parameters")
    biolog.info(parameters.showParameters())

    # Simulation Parameters
    t_start = 0.0
    t_stop = 5.0
    dt = 1.0
    biolog.warning("Using large time step dt={}".format(dt))
    time = np.arange(t_start, t_stop, dt)
    x0 = [0.0, 0.0]

    res = integrate(pendulum_integration, x0, time, args=(parameters, ))
    res.plot_state("State")
    res.plot_phase("Phase")

    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    exercise1()

