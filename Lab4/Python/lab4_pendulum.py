""" Pendulum """

import numpy as np
import biolog
from SystemParameters import PendulumParameters


def pendulum_equation(theta, dtheta, parameters=PendulumParameters()):
    """ Pendulum equation d2theta = -g/L*sin(theta)

    with:
        - theta: Angle [rad]
        - dtheta: Angular velocity [rad/s]
        - g: Gravity constant [m/s**2]
        - L: Length [m]
        - sin: np.sin
        - k1 : Spring constant of spring 1 [N/rad]
        - k2 : Spring constant of spring 2 [N/rad]
        - s_theta_ref1 : Spring 1 reference angle [rad]
        - s_theta_ref2 : Spring 2 reference angle [rad]
        - b1 : Damping constant damper 1 [N-s/rad]
        - b2 : Damping constant damper 2 [N-s/rad]
    """
    g, L, sin, k1, k2, s_theta_ref1, s_theta_ref2, b1, b2 = (
        parameters.g,
        parameters.L,
        parameters.sin,
        parameters.k1,
        parameters.k2,
        parameters.s_theta_ref1,
        parameters.s_theta_ref2,
        parameters.b1,
        parameters.b2
    )
    
    d2theta = -g/L*sin(theta) + k2*max(s_theta_ref2-theta,0) + k1*min(s_theta_ref1-theta,0)
    
    return d2theta


def pendulum_system(theta, dtheta, parameters=PendulumParameters()):
    """ Pendulum """
    return np.array([
        [dtheta],
        [pendulum_equation(theta, dtheta, parameters)]  # d2theta
    ])

