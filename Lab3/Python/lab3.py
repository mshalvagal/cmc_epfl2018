#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Lab 3 """


import numpy as np
import matplotlib.pyplot as plt

import biolog
from biopack import integrate_multiple, DEFAULT


class LeakyIntegratorParameters(object):
    """ Leaky-integrator neuron parameters """

    def __init__(self, tau, D, b, w, exp=np.exp):
        super(LeakyIntegratorParameters, self).__init__()
        self.tau = np.array(tau)  # Time constant
        self.D = np.array(D)
        self.b = np.array(b)
        self.w = np.array(w)  # Weights
        self.exp = exp  # Exponential
        return

    def __str__(self):
        """ String used when printing instantiated object """
        return self.msg()

    def list(self):
        """ Return list of parameters """
        return self.tau, self.D, self.b, self.w, self.exp

    def msg(self):
        """ Parameters information message """
        return (
            "Leaky integrator parameters:"
            "\nTau: {}"
            "\nD:   {}"
            "\nb:   {}"
            "\nw:   {}"
            "\nExp: {}"
        ).format(*self.list())


def two_li_ode(y, t, params):
    """ Derivative function of a network of 2 leaky integrator neurons

    y is the vector of membrane potentials (variable m in lecture equations)
    yd the derivative of the vector of membrane potentials
    """
    # Extract parameters
    tau, D, b, w, exp = params.list()

    # Update the firing rates:
    x = [0, 0]

    # IMPLEMENT THE DIFFERENTIAL EQUATION FOR THE MEMBRANE POTENTIAL
    # Compute the dentritic sums for both neurons
    dend_sum = [0, 0]

    # Compute the membrane potential derivative:
    yd = [0, 0]
    # biolog.debug("x: {}\ndend_sum: {}\nyd: {}".format(x, dend_sum, yd))
    return yd


def two_coupled_li_neurons(y_0, t_max, dt, params, figure="Phase"):
    """ Two mutually coupled leaky-integrator neurons with self connections """
    res = integrate_multiple(
        two_li_ode,
        y_0,
        np.arange(0, t_max, dt),
        args=(params,)
    )
    labels = ["Neuron 1", "Neuron 2"]
    res.plot_state(figure + "_state", label=False, subs_labels=labels)
    res.plot_phase(figure + "_phase", scale=0.05, label=labels)
    return res


def exercise5():
    """ Lab 3 - Exrecise 5 """
    # Fixed parameters of the neural network
    tau = [0.05, 0.05]
    D = 1

    # Additional parameters
    b = [0, 0]
    w = [[1, 0], [0, 1]]

    # All system parameters packed in object for integration
    params = LeakyIntegratorParameters(tau, D, b, w)

    # Initial conditions
    #  SET THE INITIAL CONDITIONS
    y_0 = [[0, 0]]  # Values of the membrane potentials of the two neurons

    # Integration time parameters
    # Force integration to return values at small steps for
    # better detecting Poincare maps crossings
    dt = 1e-4
    t_max = 30  # Set total simulation time

    # Integration (make sure to implement)
    biolog.warning(
        "Uncomment next line to run integration"
        " after implementing two_li_ode()"
    )
    # two_coupled_li_neurons(y_0, t_max, dt, params, "Case1")

    # Two stable fixed points and one saddle node
    biolog.warning("Implement two stable fixed points and one saddle node")

    # Limit cycle
    biolog.warning("Implement limit cycle")
    biolog.warning(u"Implement Poincare analysis of limit cycle")

    # Limit cycle (small), one stable fixed point and one saddle node
    biolog.warning(
        "Implement a system with:"
        "\n- One limit cycle (small)"
        "\n- One stable fixed point"
        "\n- One saddle node"
    )
    biolog.warning(u"Implement Poincare analysis of limit cycle")

    if DEFAULT["save_figures"] is False:
        plt.show()

    return


def main():
    """ Lab 3 exercise """
    biolog.info("Runnig exercise 5")
    exercise5()
    return


if __name__ == "__main__":
    from biopack import parse_args
    parse_args()
    main()

