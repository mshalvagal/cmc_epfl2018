""" Lab 4 : System Parameters """


class SystemParameters(object):
    """Parent class providing main attributes for other sub system
    parameters.

    """

    def __init__(self, sys_name='System'):
        super(SystemParameters, self).__init__()
        self.sys_name = sys_name

    def showParameters(self):
        raise NotImplementedError()

    def msg(self, parameters, units, endl="\n" + 4 * " "):
        """ Message """
        to_print = ("{} parameters : ".format(self.sys_name)) + endl
        for param in parameters:
            to_print += ("{} : {} [{}]".format(
                param,
                parameters[param],
                units[param]
            )) + endl
        return to_print


class MuscleParameters(SystemParameters):
    """ Muscle parameters

    with:
        Muscle Parameters:
            - l_slack : Tendon slack length [m]
            - l_opt : Contracticle element optimal fiber length [m]
            - f_max : Maximum force produced by the muscle [N]
            - v_max : Maximum velocity of the contracticle element [m/s]
            - pennation : Fiber pennation angle

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

    Or using print:

    >>> muscle_parameters = MuscleParameters()
    >>> print(muscle_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MuscleParameters, self).__init__('muscle')
        self.parameters = {}
        self.units = {}

        self.units['l_slack'] = 'm'
        self.units['l_opt'] = 'm'
        self.units['f_max'] = 'N'
        self.units['v_max'] = 'm/s'
        self.units['pennation'] = ''
        self.units['name'] = '<str>'

        self.parameters['l_slack'] = kwargs.pop('l_slack', 0.13)
        self.parameters['l_opt'] = kwargs.pop('l_opt', 0.11)
        self.parameters['f_max'] = kwargs.pop('f_max', 1500)
        self.parameters['v_max'] = kwargs.pop('v_max', 1.2)
        self.parameters['pennation'] = kwargs.pop('pennation', 1)
        self.parameters['name'] = kwargs.pop('name', 'muscle')

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
    def name(self):
        """ Name of the muscle. """
        return self.parameters['name']

    @name.setter
    def name(self, value):
        """Keyword Arguments:
           value --  Name of the muscle """
        self.parameters['name'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class MuscleJointParameters(SystemParameters):
    """ Parameters that define the interface between joint and muscle.

    with:
        Muscle Joint Parameters:
            - muscle_type : Type of muscle
                    ['mono' - Mono articular]
                    ['bi' - Bi articular]
            - r_0 : Muscle maximum moment arm across joint 1 [m]
            - joint_attach : Joint to which the muscle attaches <str>
            - theta_max : Joint 1 angle at which maximal torque is
                          generated
            - theta_ref : Jonint 1 angle at which muscle length is at its rest
                          length
            - direction : Direction of torque applied on joint 1
                          ['clockwise/cclockwise']
    Examples:

        >>> muscle_joint_parameters = (
        ...     MuscleJointParameters(m_type='mono', r_0 = 0.002)
        ... )

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Muscle Joint Parameters
        - muscle_type = 'mono'
        - r_0 = 1
        - joint_attach = 'LH_J_HIP'
        - theta_max = 0.0
        - theta_ref = 0.0
        - direction = 'clockwise'

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.m_type = 'bi' # Reassign tendon slack constant
        To assign to another variable from within the class:


        >>> example_muscle_m_type = self.m_type

    You can display the parameters using:

    >>> muscle_joint_parameters = MuscleJointParameters()
    >>> print(muscle_joint_parameters,showParameters())
    Muscle Joint parameters :
        muscle_type = 'mono'
        r_0 = 1
        joint_attach = 'LH_J_HIP'
        theta_max = 0.0
        theta_ref = 0.0
        direction = 'clockwise'

    Or using :

    >>> muscle_joint_parameters = MuscleJointParameters()
    >>> .info(muscle_joint_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MuscleJointParameters, self).__init__('muscle-joint')
        self.parameters = {}
        self.units = {}

        self.units['muscle_type'] = '<str>'
        self.units['r_0'] = 'm'
        self.units['joint_attach'] = '<str>'
        self.units['theta_max'] = 'rad'
        self.units['theta_ref'] = 'rad'
        self.units['direction'] = '<str>'

        self.parameters['muscle_type'] = kwargs.pop('muscle_type', 'mono')
        self.parameters['r_0'] = kwargs.pop('r_0', '1.')
        self.parameters['joint_attach'] = kwargs.pop('joint_attach', 'joint')
        self.parameters['theta_max'] = kwargs.pop('theta_max', 0.0)
        self.parameters['theta_ref'] = kwargs.pop('theta_ref', 0.0)
        self.parameters['direction'] = kwargs.pop('direction', 'clockwise')

    @property
    def muscle_type(self):
        """ Return if muscle is mono/bi articular. """
        return self.parameters['muscle_type']

    @muscle_type.setter
    def muscle_type(self, value):
        """ Keyword Arguments:
            value -- Set muscle type : mono/bi"""
        if (value is 'mono') or (value is 'bi'):
            self.parameters['muscle_type'] = value
        else:
            ('Invalid muscle type')

    @property
    def r_0(self):
        """Maximum muscle moment arm [m]  """
        return self.parameters['r_0']

    @r_0.setter
    def r_0(self, value):
        """ Keyword Arguments:
        value -- Maximum muscle moment arm [m]"""
        if value < 0.0:
            ('Muscle moment arm cannot be negative!')
        else:
            self.parameters['r_0'] = value

    @property
    def joint_attach(self):
        """Joint to which the muscle applies torque on."""
        return self.parameters['joint_attach']

    @joint_attach.setter
    def joint_attach(self, value):
        """ Keyword Arguments:
        value -- Joint to which the muscle applies torque on"""
        self.parameters['joint_attach'] = value

    @property
    def theta_max(self):
        """Joint angle at which maximum muscle torque is applied [rad]  """
        return self.parameters['theta_max']

    @theta_max.setter
    def theta_max(self, value):
        """ Keyword Arguments:
        value -- Joint angle at which maximum muscle torque is applied [rad]"""
        self.parameters['theta_max'] = value

    @property
    def theta_ref(self):
        """Joint angle muscle length is at its rest length [rad]  """
        return self.parameters['theta_ref']

    @theta_ref.setter
    def theta_ref(self, value):
        """ Keyword Arguments:
        value -- Joint angle muscle length is at its rest length [rad]  """
        self.parameters['theta_ref'] = value

    @property
    def direction(self):
        """Direction of muscle torque. <str>
            'clockwise'
            'cclockwise'. """
        return self.parameters['direction']

    @direction.setter
    def direction(self, value):
        """ Keyword Arguments:
        value -- Direction of muscle torque. <str>
            'clockwise'
            'cclockwise'. """
        self.parameters['direction'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class JointParameters(SystemParameters):
    """ Parameters that define the interface between joint.

    with:
        Joint Parameters:
            - name : Name of the joint <str>
            - theta_max : Maximum allowed joint angle rotation
            - theta_min : Minimum allowed joint angle rotation
            - reference_angle : Joint offset from the simulation anlge
            - joint_type : Moment arm computation type <str> [GEYER/CONSTANT]
    Examples:

        >>> joint_parameters = JointParameters(name='Joint1',
                                                            theta_min=0.0)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Joint Parameters
        - name = 'joint'
        - theta_max = 1.5707
        - theta_min = -1.5707
        - reference_angle = 0.0
        - joint_type = 'GEYER'

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.name = 'Joint' # Reassign tendon slack constant
        To assign to another variable from within the class:


        >>> example_joint_name = self.name

    You can display the parameters using:

    >>> joint_parameters = JointParameters()
    >>> print(joint_parameters,showParameters())
    Joint parameters :
        joint parameters :
            theta_min : -1.5707 [rad]
            joint_type : GEYER [<str>]
            reference_angle : 0.0 [rad]
            name : joint [<str>]
            theta_max : 1.5707 [rad]


    Or using :

    >>> joint_parameters = JointParameters()
    >>> .info(joint_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(JointParameters, self).__init__('joint')
        self.parameters = {}
        self.units = {}

        self.units['name'] = '<str>'
        self.units['theta_max'] = 'rad'
        self.units['theta_min'] = 'rad'
        self.units['reference_angle'] = 'rad'
        self.units['joint_type'] = '<str>'

        self.parameters['name'] = kwargs.pop('name', 'joint')
        self.parameters['theta_max'] = kwargs.pop('theta_max', 1.5707)
        self.parameters['theta_min'] = kwargs.pop('theta_min', -1.5707)
        self.parameters['reference_angle'] = kwargs.pop('reference_angle', 0.0)
        self.parameters['joint_type'] = kwargs.pop('joint_type', 'GEYER')

    @property
    def name(self):
        """Name of the joint.  """
        return self.parameters['name']

    @name.setter
    def name(self, value):
        """Keyword Arguments:
           value --  Name of the joint <str> """
        self.parameters['name'] = value

    @property
    def theta_max(self):
        """Maximum allowed joint rotation.  """
        return self.parameters['theta_max']

    @theta_max.setter
    def theta_max(self, value):
        """Keyword Arguments:
           value --  Maximum allowed joint rotation [rad] """
        self.parameters['theta_max'] = value

    @property
    def theta_min(self):
        """Minimum allowed joint rotation.  """
        return self.parameters['theta_min']

    @theta_min.setter
    def theta_min(self, value):
        """Keyword Arguments:
           value --  Minimum allowed joint rotation [rad] """
        self.parameters['theta_min'] = value

    @property
    def reference_angle(self):
        """Joint reference/offset in angle measurement.  """
        return self.parameters['reference_angle']

    @theta_min.setter
    def theta_min(self, value):
        """Keyword Arguments:
           value --  Joint reference/offset in angle measurement [rad] """
        self.parameters['reference_angle'] = value

    @property
    def joint_type(self):
        """Type of joint moment arm computation  """
        return self.parameters['joint_type']

    @joint_type.setter
    def joint_type(self, value):
        """Keyword Arguments:
           value --  Type of joint moment arm computation """
        self.parameters['joint_type'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


if __name__ == '__main__':
    M = MuscleParameters()
    print((M.showParameters()))

    MJ = MuscleJointParameters()
    print((MJ.showParameters()))

    J = JointParameters()
    print((J.showParameters()))
