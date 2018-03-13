""" Pendulum """

import numpy as np
import biolog


class PendulumParameters(object):
    """ Pendulum system parameters

    with:
        - g: Gravity constant [m/s**2]
        - L: Length [m]
        - d: Damping coefficient
        - sin: Sine function
        - dry: Use dry friction (bool: True or False)

    Examples:

        >>> pendulum_parameters = PendulumParameters(g=9.81, L=0.1)
        >>> pendulum_parameters = PendulumParameters(d=0.3, dry=True)

    Note that not giving arguments to instanciate the object will result in the
    following default values:

        - g = 9.81
        - L = 1.
        - d = 0.3
        - sin = np.sin
        - dry = False

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.g = 9.81 # Reassign gravity constant

        To assign to another variable from within the class:

        >>> example_g = self.g

    To call the parameters from outside the class, such as after instatiation
    similarly to the example above:

        To assign a new value to the object variable from outside the class:

        >>> pendulum_parameters = PendulumParameters(L=1.0)
        >>> pendulum_parameters.L = 0.3 # Reassign length

        To assign to another variable from outside the class:

        >>> pendulum_parameters = PendulumParameters()
        >>> example_g = pendulum_parameters.g # example_g = 9.81

    You can display the parameters using:

    >>> pendulum_parameters = PendulumParameters()
    >>> print(pendulum_parameters)
    Pendulum system parameters:
        g: 9.81 [m/s**2]
        L: 1.0 [m]
        d: 0.3
        sin: <ufunc 'sin'>
        dry: False

    Or using biolog:

    >>> pendulum_parameters = PendulumParameters()
    >>> biolog.info(pendulum_parameters)
    """

    def __init__(self, **kwargs):
        super(PendulumParameters, self).__init__()
        self.g = kwargs.pop("g", 9.81)  # Gravity constant
        self.L = kwargs.pop("L", 1.)  # Length
        self.d = kwargs.pop("d", 0.3)  # damping
        self.sin = kwargs.pop("sin", np.sin)  # Sine function
        self.dry = kwargs.pop("dry", False)  # Use dry friction (True or False)
        biolog.info(self)
        return

    def __str__(self):
        return self.msg()

    def msg(self, endl="\n" + 4 * " "):
        """ Message """
        return (
            "Pendulum system parameters:"
            + 5 * (endl + "{}: {} {}")
        ).format(
            "g", self.g, "[m/s**2]",
            "L", self.L, "[m]",
            "d", self.d, "",
            "sin", self.sin, "",
            "dry", self.dry, ""
        )

    @classmethod
    def from_list(cls, parameter_list):
        """ Generate object from list of arguments

        Order is:
        0) gravity (float)
        1) Length (float)
        2) Damping (float)
        3) Sine function (function)
        4) Dry friction (bool)
        """
        size = len(parameter_list)
        keys = ["g", "L", "d", "sin", "dry"]
        return cls(**{
            key: parameter_list[i]
            for i, key in enumerate(keys)
            if size > i
        })


def pendulum_equation(theta, dtheta, time=0, parameters=None):
    """ Pendulum equation

    with:
        - theta: Angle [rad]
        - dtheta: Angular velocity [rad/s]
        - g: Gravity constant [m/s**2]
        - L: Length [m]
        - d: Damping coefficient []
    """
    if parameters is None:
        parameters = PendulumParameters()
        biolog.warning(
            "Parameters not given, using defaults\n{}".format(parameters)
        )
    g, L, d, sin, dry = (
        parameters.g,
        parameters.L,
        parameters.d,
        parameters.sin,
        parameters.dry
    )
    if dry:
        d2theta = -np.sin(theta)*g/L - d*np.sign(dtheta)
    else:
        d2theta = -np.sin(theta)*g/L - d*dtheta
    return d2theta


def pendulum_system(theta, dtheta, time=0, parameters=None):
    """ Pendulum """
    return np.array([
        [dtheta],
        [pendulum_equation(theta, dtheta, time, parameters)]  # d2theta
    ])

