""" Lab 4 : System Parameters """

import numpy as np
import biolog


class SystemParameters(object):
    """Parent class providing main attributes for other sub system
    parameters.

    """

    def __init__(self, name='System'):
        super(SystemParameters, self).__init__()
        self. name = name

    def showParameters(self):
        raise NotImplementedError()

    def msg(self, parameters, units, endl="\n" + 4 * " "):
        """ Message """
        to_print = ("{} parameters : ".format(self.name)) + endl
        for param in parameters:
            to_print += ("{} : {} [{}]".format(param,
                                               parameters[param], units[param])) + endl
        return to_print


class PendulumParameters(SystemParameters):
    """ Pendulum parameters

    with:
    ----
        Pendulum Parameters:
            - g: Gravity constant [m/s**2]
            - L: Length [m]
            - sin: Sine function
            - mass: mass of the pendulum [kg]
            - Inertia: intertia of the pendulum [kg-m**2]
            - theta_max: Maximum pendulum angle
            - theta_min: Minimum pendulum angle

    Examples:

        >>> pendulum_parameters = PendulumParameters(g=9.81, L=0.1)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        Pendulum parameters :
            theta_min : -1.57079632679 [rad]
            g : 9.81 [N-m/s2]
            I : 0.333333333333 [kg-m**2]
            L : 0.5 [m]
            mass : 1.0 [kg]
            sin : <ufunc 'sin'> []
            theta_max : 1.57079632679 [rad]


    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.g = 9.81 # Reassign gravity constant

        To assign to another variable from within the class:

        >>> example_g = self.g

    To call the parameters from outside the class, such as after instatiation
    similarly to the example above:

        To assign a new value to the object variable from outside the class:

        >>> pendulum_parameters = SystemParameters(L=1.0)
        >>> pendulum_parameters.L = 0.3 # Reassign length

        To assign to another variable from outside the class:

        >>> pendulum_parameters = SystemParameters()
        >>> example_g = pendulum_parameters.g # example_g = 9.81

    You can display the parameters using:

    >>> pendulum_parameters = SystemParameters()
    >>> print(pendulum_parameters.showParameters())
    Pendulum parameters :
        theta_min : -1.57079632679 [rad]
        g : 9.81 [N-m/s2]
        I : 0.333333333333 [kg-m**2]
        L : 0.5 [m]
        mass : 1.0 [kg]
        sin : <ufunc 'sin'> []
        theta_max : 1.57079632679 [rad]

    Or using biolog:

    >>> pendulum_parameters = SystemParameters()
    >>> biolog.info(system_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(PendulumParameters, self).__init__('Pendulum')

        self.parameters = {}
        self.units = {}

        self.units['g'] = 'N-m/s2'
        self.units['L'] = 'm'
        self.units['sin'] = ''
        self.units['theta_max'] = 'rad'
        self.units['theta_min'] = 'rad'
        self.units['mass'] = 'kg'
        self.units['I'] = 'kg-m**2'

        # Pendulum parameters
        self.parameters['g'] = kwargs.pop("g", 9.81)  # Gravity constant
        self.parameters['L'] = kwargs.pop("L", 1.)  # Length
        self.parameters['mass'] = kwargs.pop("mass", 1.)  # Pendulum Mass
        # Inertia
        self.setInertia()
        self.parameters['sin'] = kwargs.pop("sin", np.sin)  # Sine function
        self.parameters['theta_max'] = kwargs.pop(
            "theta_max", np.pi / 2.)  # Pendulum Maximum angle
        self.parameters['theta_min'] = kwargs.pop(
            "theta_min", -np.pi / 2.)  # Pendulum Minimum angle
        return

    @property
    def g(self):
        """ Get the value of gravity in the system. [N-m/s2]
        Default is 9.81 """
        return self.parameters['g']

    @g.setter
    def g(self, value):
        """ Keyword Arguments:
        value -- Set the value of gravity [N-m/s2] """
        self.parameters['g'] = value
        biolog.info(
            'Changed gravity to {} [N-m/s2]'.format(self.parameters['g']))

    @property
    def L(self):
        """ Get the value of pendulum length. [m]
        Default is 1.0"""
        return self.parameters['L']

    @L.setter
    def L(self, value):
        """ Keyword Arguments:
        value -- Set the value of pendulum's length [m] """
        self.parameters['L'] = value
        self.setInertia()
        biolog.info(
            'Changed pendulum length to {} [m]'.format(self.parameters['L']))

    @property
    def mass(self):
        """Pendulum mass  """
        return self.parameters['mass']

    @mass.setter
    def mass(self, value):
        """Keyword Arguments:
           value --  Set the value of pendulum's mass [kg] """
        self.parameters['mass'] = value
        self.setInertia()
        biolog.info('Changed pendulum mass to {} [kg]'.format(
            self.parameters['mass']))

    @property
    def I(self):
        """Pendulum I  """
        return self.parameters['I']

    @I.setter
    def I(self, _):
        raise Exception(
            'Cannot set Inertia. Use mass and length properties to change inertia')

    def setInertia(self):
        """Keyword Arguments:
           value --  Set the value of pendulum's I [kg] """
        self.parameters['I'] = self.mass * self.L**2 / 3.

    @property
    def sin(self):
        """ Get the sine function."""
        return self.parameters['sin']

    @property
    def theta_max(self):
        """ Joint maximum rotation angle  """
        return self.parameters['theta_max']

    @theta_max.setter
    def theta_max(self, value):
        """ Keyword Arguments:
            value -- Joint maximum rotation angle """
        self.parameters['theta_max'] = value

    @property
    def theta_min(self):
        """ Joint minimum rotation angle.  """
        return self.parameters['theta_min']

    @theta_min.setter
    def theta_min(self, value):
        """Keyword Arguments:
           value -- Joint minimum rotation angle """
        self.parameters['theta_min'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class MuscleParameters(SystemParameters):
    """ Muscle-Joint parameters

    with:
        Muscle Parameters:
            - l_slack : Tendon slack length [m]
            - l_opt : Contracticle element optimal fiber length [m]
            - f_max : Maximum force produced by the muscle [N]
            - v_max : Maximum velocity of the contracticle element [m/s]
            - pennation : Fiber pennation angle
            - reference_angle : Joint reference angle for the muscle

    Examples:

        >>> muscle_parameters = MuscleParameters(l_slack=0.2, l_opt=0.1)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Muscle Parameters
        - l_slack = 0.13
        - l_opt = 0.11
        - f_max = 1500
        - v_max = 1.2
        - pennation = 1.
        - reference_angle = 0.0

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.l_slack = 0.2 # Reassign tendon slack constant

        To assign to another variable from within the class:

        >>> example_l_slack = self.l_slack

    You can display the parameters using:

    >>> muscle_joint_parameters = MuscleParameters()
    >>> print(muscle_parameters,showParameters())
    Muscle parameters :
            f_max : 1500 [N]
            v_max : 1.2 [m/s]
            pennation : 1 []
            l_slack : 0.13 [m]
            l_opt : 0.11 [m]
            reference_angle : 0.0 [rad]

    Or using biolog:

    >>> muscle_parameters = MuscleParameters()
    >>> biolog.info(muscle_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MuscleParameters, self).__init__('Muscle')
        self.parameters = {}
        self.units = {}

        self.units['l_slack'] = 'm'
        self.units['l_opt'] = 'm'
        self.units['f_max'] = 'N'
        self.units['v_max'] = 'm/s'
        self.units['pennation'] = ''
        self.units['reference_angle'] = 'rad'

        self.parameters['l_slack'] = kwargs.pop('l_slack', 0.13)
        self.parameters['l_opt'] = kwargs.pop('l_opt', 0.11)
        self.parameters['f_max'] = kwargs.pop('f_max', 1500)
        self.parameters['v_max'] = kwargs.pop('v_max', 1.2)
        self.parameters['pennation'] = kwargs.pop('pennation', 1)
        self.parameters['reference_angle'] = kwargs.pop('reference_angle', 0.0)

    @property
    def l_slack(self):
        """ Muscle Tendon Slack length [m]  """
        return self.parameters['l_slack']

    @l_slack.setter
    def l_slack(self, value):
        """ Keyword Arguments:
            value -- Muscle Tendon Slack Length [m]"""
        self.parameters['l_slack'] = value

    @property
    def l_opt(self):
        """ Muscle Optimal Fiber Length [m]  """
        return self.parameters['l_opt']

    @l_opt.setter
    def l_opt(self, value):
        """ Keyword Arguments:
        value -- Muscle Optimal Fiber Length [m]"""
        self.parameters['l_opt'] = value

    @property
    def f_max(self):
        """ Maximum tendon force produced by the muscle [N]  """
        return self.parameters['f_max']

    @f_max.setter
    def f_max(self, value):
        """ Keyword Arguments:
        value -- Maximum tendon force produced by the muscle [N]"""
        self.parameters['f_max'] = value

    @property
    def v_max(self):
        """ Maximum velocity of the contractile element [m/s]  """
        return self.parameters['v_max']

    @v_max.setter
    def v_max(self, value):
        """ Keyword Arguments:
        value -- Maximum velocity of the contractile element [m/s] """
        self.parameters['v_max'] = value

    @property
    def pennation(self):
        """ Muscle fiber pennation angle  """
        return self.parameters['pennation']

    @pennation.setter
    def pennation(self, value):
        """ Keyword Arguments:
            value -- Muscle fiber pennation angle """
        self.parameters['pennation'] = value

    @property
    def reference_angle(self):
        """ Muscle Joint Reference angle  """
        return self.parameters['reference_angle']

    @reference_angle.setter
    def reference_angle(self, value):
        """ Keyword Arguments:
            value -- Muscle joint reference angle """
        self.parameters['reference_angle'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class NetworkParameters(SystemParameters):
    """ Network parameters

    with:
        Network Parameters:
            - tau : Array of time constants for each neuron [s]
            - D : Sigmoid constants for each neuron
            - b : Array of bias for each neuron
            - w : Weight matrix for network connections
            - exp : Exponential function <exp>

    Examples:

        >>> network_parameters = NetworkParameters(tau=[0.02, 0.02, 0.1, 0.1], D=1.)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Neuron Parameters
        - tau = [0.02, 0.02, 0.1, 0.1]
        - D = 1
        - b = [3.0, 3.0, -3.0, -3.0]
        - w = [[0., 1., 1., 1.],
               [1., 0., 1., 1.],
               [1., 1., 0., 1.],
               [1., 1., 1., 0.]]
        - exp = np.exp

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.tau[0] = 0.01  # Reassign tendon slack constant

        To assign to another variable from within the class:

        >>> example_tau = self.tau

    You can display the parameters using:

    >>> network_parameters = NetworkParameters()
    >>> print(network_parameters,showParameters())
    Network parameters :
        tau = [0.02, 0.02, 0.1, 0.1]
        D = 1
        b = [3.0, 3.0, -3.0, -3.0]
        w = [[0., 1., 1., 1.],
             [1., 0., 1., 1.],
             [1., 1., 0., 1.],
             [1., 1., 1., 0.]]
        exp = np.exp

    Or using biolog:

    >>> network_parameters = NetworkParameters()
    >>> biolog.info(network_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(NetworkParameters, self).__init__('network')
        self.parameters = {}
        self.units = {}

        self.units['tau'] = 's'
        self.units['D'] = '-'
        self.units['b'] = '-'
        self.units['w'] = '-'
        self.units['exp'] = '<exp>'

        self.parameters['tau'] = kwargs.pop(
            'tau', np.array([0.02, 0.02, 0.1, 0.1]))
        self.parameters['D'] = kwargs.pop('D', 1.)
        self.parameters['b'] = kwargs.pop('b', np.array([3., 3., -3., -3.]))
        weight_ = np.ones((4, 4))
        np.fill_diagonal(weight_, 0.)
        self.parameters['w'] = kwargs.pop(
            'w', weight_)
        self.parameters['exp'] = kwargs.pop('exp', np.exp)

    @property
    def tau(self):
        """ Time constants for neurons in the network  """
        return self.parameters['tau']

    @tau.setter
    def tau(self, value):
        """ Keyword Arguments:
            value -- Time constants for neurons in the network"""
        self.parameters['tau'] = value

    @property
    def D(self):
        """Sigmoid constant  """
        return self.parameters['D']

    @D.setter
    def D(self, value):
        """Keyword Arguments:
           value --  Sigmoid constant """
        self.parameters['D'] = value

    @property
    def b(self):
        """Bias for neurons in the network  """
        return self.parameters['b']

    @b.setter
    def b(self, value):
        """Keyword Arguments:
           value --  Bias for neurons in the network """
        self.parameters['b'] = value

    @property
    def w(self):
        """ weight matrix for the network  """
        return self.parameters['w']

    @w.setter
    def w(self, value):
        """Keyword Arguments:
           value --  weight matrix for the network """
        self.parameters['w'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)

    @property
    def exp(self):
        """Exponential"""
        return self.parameters['exp']


if __name__ == '__main__':
    P = PendulumParameters(g=9.81, L=1.)
    print(P.showParameters())

    M = MuscleParameters()
    print(M.showParameters())

    N = NetworkParameters()
    print(N.showParameters())

