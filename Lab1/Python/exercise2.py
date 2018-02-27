""" Lab 1 - Exercise 2 """

import numpy as np
import matplotlib.pyplot as plt

from biopack import integrate, DEFAULT
import biolog


def ode(x, _=None, A=np.eye(2)):
    """ System x_dot = A*x """
    return np.dot(A, x)


def integration(x0, time, A, name, **kwargs):
    """ System integration """
    labels = kwargs.pop("label", ["State {}".format(i) for i in range(2)])
    sys_int = integrate(ode, x0, time, args=(A,))
    sys_int.plot_state("{}_state".format(name), labels)
    sys_int.plot_phase("{}_phase".format(name))
    return


def exercise2():
    """ Exercise 2 """
    biolog.info("Running exercise 2")

    # System definition
    biolog.warning("Proper matrix A must be implemented")
    A = np.array([[1, 0], [0, 1]])
    time_total = 10
    time_step = 0.01
    x0, time = [0, 1], np.arange(0, time_total, time_step)

    # Normal run
    biolog.warning("System integration must be implemented")
    # integration(x0, time, A, "example")

    # Stable point (Optional)
    biolog.warning("Stable point integration must be implemented")

    # Periodic
    biolog.warning("Periodic system must be implemented")

    # Plot
    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == "__main__":
    from biopack import parse_args
    parse_args()
    exercise2()