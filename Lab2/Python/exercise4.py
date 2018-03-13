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
    params = HopfParameters(1.0, np.pi)
    time = np.arange(0, 300, 0.01)  # Simulation time
    x0 = [0.2, 0.3]
    res = integrate(hopf_equation, x0, time, args=(params,))
    res.plot_phase("Phase")

    temp = np.size(time)/50
    x0 = [0.1, 0.0]
    for i in range(50):
        res = integrate(hopf_equation, x0, time[temp*i:temp*(i+1)], args=(params,))
        x0 = [-1.0+2.0*np.random.random(),-1.0+2.0*np.random.random()]
#        res.plot_phase("Phase_perturbation")
    
    return


def coupled_hopf_ocillator():
    """ 4b - Coupled Hopf oscillator simulation """
    biolog.warning("Coupled Hopf oscillator must be implemented")
    params = CoupledHopfParameters([1.0,np.pi],[1.2,np.pi*0.33],[1,1])
    time = np.arange(0, 100, 0.01)  # Simulation time
    x0 = [0.2, 0.3, 0.2, 0.3]
    res = integrate(coupled_hopf_equation, x0, time, args=(params,))
    res.plot_state("State")
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

