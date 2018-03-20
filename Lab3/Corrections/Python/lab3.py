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
    x = 1./(1+exp(-D*(y+b)))

    # IMPLEMENT THE DIFFERENTIAL EQUATION FOR THE MEMBRANE POTENTIAL
    # Compute the dentritic sums for both neurons
    dend_sum = np.dot(w, x)

    # Compute the membrane potential derivative:
    yd = (dend_sum-y)/tau
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
    res.plot_state(figure+"_state", label=False, subs_labels=labels)
    res.plot_phase(figure+"_phase", scale=0.05, label=labels)
    return res


def poincare_crossings(res, threshold, crossing_index, figure):
    """ Study poincaré crossings """
    ci = crossing_index

    # Extract state of first trajectory
    state = np.array(res.state[0])

    # Crossing index (index corrsponds to last point before crossing)
    idx = np.argwhere(np.diff(np.sign(state[:, ci] - threshold)) < 0)
    # biolog.debug("Indices:\n{}".format(idx))  # Show crossing indices

    # Linear interpolation to find crossing position on threshold
    # Position before crossing
    pos_pre = np.array([state[index[0], :] for index in idx])
    # Position after crossing
    pos_post = np.array([state[index[0]+1, :] for index in idx])
    # Position on threshold
    pos_treshold = [
        (
            (threshold - pos_pre[i, 1])/(pos_post[i, 1] - pos_pre[i, 1])
        )*(
            pos_post[i, 0] - pos_pre[i, 0]
        ) + pos_pre[i, 0]
        for i, _ in enumerate(idx)
    ]

    # Plot
    # Figure limit cycle variance
    plt.figure(figure)
    plt.plot(pos_treshold, "o-")
    val_min = np.sort(pos_treshold)[2]
    val_max = np.sort(pos_treshold)[-2]
    bnd = 0.3*(val_max - val_min)
    plt.ylim([val_min-bnd, val_max+bnd])
    plt.xlabel(u"Number of Poincaré section crossings")
    plt.ylabel("Value for Neuron 1 (Neuron 2 = {})".format(threshold))
    plt.grid(True)

    # Figure limit cycle
    plt.figure(figure+"_phase")
    plt.plot([val_min-0.3, val_max+0.3], [threshold]*2, "gx--")
    for pos in pos_treshold:
        plt.plot(pos, threshold, "ro")

    # Save plots if option activated
    if DEFAULT["save_figures"] is True:
        from biopack.plot import save_figure
        save_figure(figure)
        save_figure(figure+"_phase")

        # Zoom on limit cycle
        plt.figure(figure+"_phase")
        plt.xlim([val_min-bnd, val_max+bnd])
        plt.ylim([threshold-1e-7, threshold+1e-7])
        save_figure(figure=figure+"_phase", name=figure+"_phase_zoom")

    return idx


def exercise5():
    """ Lab 3 - Exrecise 5 """
    # Fixed parameters of the neural network
    tau = [0.05, 0.05]
    D = 1


    # Integration time parameters
    # Force integration to return values at small steps for
    # better detecting Poincare maps crossings
    dt = 1e-4

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


    # Two stable fixed points and one saddle node
    t_max = 2
    b = [-3.4, -2.5]
    w = [[5.25, 1], [-1, 5.25]]
    y_0 = [
        [i, j]
        for i in np.linspace(0, 3, 2)
        for j in np.linspace(-2, 6, 15)
    ]
    params = LeakyIntegratorParameters(tau, D, b, w)
    two_coupled_li_neurons(y_0, t_max, dt, params, "Case1")

    # Limit cycle
    t_max = 3
    b = [-2.75, -1.75]
    w = [[4.5, 1], [-1, 4.5]]
    y_0 = [
        [i, j]
        for i in np.linspace(0, 6, 2)
        for j in np.linspace(-1, 4, 5)
    ] + [[2.7, 1.7]]
    params = LeakyIntegratorParameters(tau, D, b, w)
    two_coupled_li_neurons(y_0, t_max, dt, params, "Case2")
    y_0 = [[2.7, 1.7]]
    t_max = 30
    res = two_coupled_li_neurons(y_0, t_max, dt, params, "Case2_cross")
    poincare_crossings(res, 2.5, 1, "Case2_cross")

    # Limit cycle (small), one stable fixed point and one saddle node
    t_max = 10
    b = [-3.233, -1.75]
    w = [[5.5, 1], [-1, 5.5]]
    y_0 = [
        [i, j]
        for ii, i in enumerate(np.linspace(3, 7, 2))
        for j in np.linspace(-4, 6, 15 if ii > 0 else 5)
    ]
    params = LeakyIntegratorParameters(tau, D, b, w)
    two_coupled_li_neurons(y_0, t_max, dt, params, "Case3")
    y_0 = [[5, 0]]
    t_max = 30
    res = two_coupled_li_neurons(y_0, t_max, dt, params, "Case3_cross")
    poincare_crossings(res, 0.5, 1, "Case3_cross")


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