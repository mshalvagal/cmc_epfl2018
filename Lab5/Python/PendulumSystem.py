""" Pendulum """

import numpy as np
import biolog
from SystemParameters import PendulumParameters


class Pendulum(object):
    """Pendulum model main class."""

    def __init__(self, parameters=PendulumParameters()):
        super(Pendulum, self).__init__()
        self.origin = np.array([0.0, 0.0])
        self.theta = 0.0
        self.dtheta = 0.0
        self.parameters = parameters
        return

    def derivative(self, state, time, *args, **kwargs):
        """Computes the derivative of the pendulum eqn for integration.

        Parameters:
        -----------
        state: np.array
            Position and Velocity of the pendulum
        time: float
            Current time
        *args: tuple
            External arguments to the system
        **kwargs: dict
            External arguments to the system

        Returns:
        --------
        derivative: np.array
            Return the current Acceleration and Velocity of the pendulum"""

        # YOU CAN ADD PERTURBATIONS TO THE PENDULUM MODEL HERE

        # External torque applied to the pendulum
        torque = args[0]
        return pendulum_system(
            state[0],
            state[1],
            torque,
            self.parameters
        )[:, 0]

    def pose(self):
        """Compute the full pose of the pendulum.

        Returns:
        --------
        pose: np.array
            [origin, center-of-mass]"""
        return np.array(
            [self.origin,
             self.origin + self.link_pose()])

    def link_pose(self):
        """ Position of the pendulum center of mass.

        Returns:
        --------
        link_pose: np.array
            Returns the current pose of pendulum COM"""

        return self.parameters.L * np.array([
            np.sin(self.theta),
            -np.cos(self.theta)])

    @property
    def state(self):
        """ Get the pendulum state  """
        return [self.theta, self.dtheta]

    @state.setter
    def state(self, value):
        """"Set the state of the pendulum.

        Parameters:
        -----------
        value: np.array
            Position and Velocity of the pendulum"""

        self.theta = value[0]
        self.dtheta = value[1]


def pendulum_equation(theta, dtheta, torque, parameters):
    """ Pendulum equation d2theta = -mgL*sin(theta)/I + torque/I

    with:
        - theta: Angle [rad]
        - dtheta: Angular velocity [rad/s]
        - g: Gravity constant [m/s**2]
        - L: Length [m]
        - mass: Mass [kg]
        - I: Inertia [kg-m**2]
        - sin: np.sin
    """
    g, L, sin, mass, I = (
        parameters.g,
        parameters.L,
        parameters.sin,
        parameters.mass,
        parameters.I
    )
    return 0


def pendulum_system(theta, dtheta, torque, parameters):
    """ Pendulum."""

    return np.array([
        [dtheta],
        [pendulum_equation(theta, dtheta, torque, parameters)]  # d2theta
    ])

