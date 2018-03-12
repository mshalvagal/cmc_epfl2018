""" Lab 2 """

import numpy as np
import matplotlib.pyplot as plt

from biopack import integrate, DEFAULT, parse_args
import biolog

from ex3_pendulum import PendulumParameters, pendulum_system
from ex3_analyse import fixed_points  # Optional


DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]


def pendulum_integration(state, time=None, parameters=None):
    """ Function for system integration """
    return pendulum_system(state[0], state[1], time, parameters)[:, 0]


def pendulum_perturbation(state, time=None, parameters=None):
    """ Function for system integration with perturbation """
    if time > 10 and time < 10.1:
        state[0], state[1] = 0.5, 0.0
    return pendulum_system(state[0], state[1], time, parameters)[:, 0]


def evolution_cases(time):
    """ Normal simulation """
    biolog.info("Evolution with basic paramater")
    x0_cases = [
        ["Normal", [0.1, 0]],
        ["Stable", [0.0, 0.0]],
        ["Unstable", [np.pi, 0.0]],
        ["Multiple loops", [0.1, 10.0]]
    ]
    title = "{} case {} (x0={})"
    parameters = PendulumParameters()
    for name, x0 in x0_cases:
        res = integrate(pendulum_integration, x0, time, args=(parameters,))
        res.plot_state(title.format(name, "state", x0))
        res.plot_phase(title.format(name, "phase", x0))
    return


def evolution_no_damping(x0, time):
    """ No damping simulation """
    biolog.info("Evolution with no damping")
    parameters = PendulumParameters(d=0.0)
    biolog.info(parameters)
    title = "{} without damping (x0={})"
    res = integrate(pendulum_integration, x0, time, args=(parameters,))
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))
    return


def evolution_perturbation(x0, time):
    """ Perturbation and no damping simulation """
    biolog.info("Evolution with perturbations")
    parameters = PendulumParameters(d=0.0)
    biolog.info(parameters)
    title = "{} with perturbation (x0={})"
    res = integrate(
        pendulum_perturbation, x0, time, args=(parameters,)
    )
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))
    return


def evolution_dry(x0, time):
    """ Dry friction simulation """
    biolog.info("Evolution with dry friction")
    parameters = PendulumParameters(d=0.03, dry=True)
    biolog.info(parameters)
    title = "{} with dry friction (x0={})"
    res = integrate(pendulum_integration, x0, time, args=(parameters,))
    res.plot_state(title.format("State", x0))
    res.plot_phase(title.format("Phase", x0))
    return


def exercise3():
    """ Exercise 3 """
    fixed_points()  # Optional
    parameters = PendulumParameters()  # Checkout pendulum.py for more info
    biolog.info(parameters)
    # Simulation parameters
    time = np.arange(0, 30, 0.01)  # Simulation time
    x0 = [0.1, 0.0]  # Initial state


    # Evolutions
    evolution_cases(time)
    x0 = [0.1, 0]
    evolution_no_damping(x0, time)
    evolution_perturbation(x0, time)
    evolution_dry(x0, time)

    # Show plots of all results
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    parse_args()
    exercise3()