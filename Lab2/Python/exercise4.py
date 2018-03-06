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
    biolog.warning("Hopf oscillator must be implemented")
    # params = HopfParameters()
    return


def coupled_hopf_ocillator():
    """ 4b - Coupled Hopf oscillator simulation """
    biolog.warning("Coupled Hopf oscillator must be implemented")
    # param = CoupledHopfParameters()
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

