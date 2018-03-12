""" Lab 2 - Exercise 4 """

import numpy as np
import matplotlib.pyplot as plt

from biopack import integrate, integrate_multiple, DEFAULT, parse_args
import biolog

from ex4_hopf import (
    HopfParameters,
    CoupledHopfParameters,
    hopf_equation,
    coupled_hopf_equation
)


def hopf_ocillator():
    """ 4a - Hopf oscillator simulation """
    params = HopfParameters(mu=1., omega=1.0)
    time = np.arange(0, 30, 0.01)
    title = "Hopf oscillator {} (x0={})"
    label = ["x0", "x1"]
    x0_list = [[1e-3, 1e-3], [1.0, 1.0]]
    for x0 in x0_list:
        hopf = integrate(hopf_equation, x0, time, args=(params,))
        hopf.plot_state(title.format("state", x0), label)
    hopf_multiple = integrate_multiple(
        hopf_equation,
        x0_list,
        time,
        args=(params,)
    )
    hopf_multiple.plot_phase(title.format("phase", x0_list), label=label)
    return


def coupled_hopf_ocillator():
    """ 4b - Coupled Hopf oscillator simulation """
    param_set = [
        CoupledHopfParameters(
            mu=[1., 1.],
            omega=[1.0, 1.2],
            k=[-0.5, -0.5]
        ),
        CoupledHopfParameters(
            mu=[1., 1.],
            omega=[1.0, 2.0],
            k=[-0.5, -0.5]
        )
    ]
    for param in param_set:
        for x0 in [[0.0, 1.0, 0.0, 1.0], [1e-3]*4]:
            time = np.arange(0, 30, 0.01)
            hopf = integrate(coupled_hopf_equation, x0, time, args=(param,))
            title = "Coupled Hopf oscillator {} (x0={}, {})"
            label = ["x0", "x1", "x2", "x3"]
            label_angle = ["angle0", "angle1"]
            hopf.plot_state(title.format("state", x0, param), label, n_subs=2)
            hopf.plot_angle(title.format("angle", x0, param), label_angle)
    return


def exercise4():
    """ Exercise 4 """
    hopf_ocillator()
    coupled_hopf_ocillator()
    # Show plots of all results
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    parse_args()
    exercise4()