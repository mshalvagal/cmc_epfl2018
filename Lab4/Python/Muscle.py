import numpy as np


class Muscle(object):
    """This class implements the muscle model.
    The muscle model is based on the hill-type muscle model.
    """
    # Default Muscle Parameters

    c = np.log(0.05)  # pylint: disable=no-member
    N = 1.5
    K = 5.0
    tau_act = 0.01  # Time constant for the activation function
    F_per_m2 = 300000  # Force per m2 of muscle PCSA
    density = 1060

    def __init__(self, parameters):
        """This function initializes the muscle model.
        A default muscle name is given as muscle

        Parameters
        ----------
        parameters : <MuscleParameters>
            Instance of MuscleParameters class

        Returns:
        -------
        Muscle : <Muscle>
            Returns an instance of class Muscle

        Attributes:
        ----------
        l_MTC : float
            Length of Muscle Tendon Complex
        l_slack : float
            Tendon slack length
        l_opt : float
            Optimal fiber length
        l_CE : float
            Length of contracticle element
        v_CE : float
            Velocity of contractile element
        deltaLength : float
            Change in Muscle Tendon length
        
        : float
            Active force generated by the muscle
        passiveForce : float
            Passive force generated by the muscle
        force : float
            Sum of Active and Passive forces
        tendonForce : float
            Force generated by the muscle tendon
        stim : float
            Muscle stimulation.

        Methods:
        --------
        step : func
            Integrates muscle state by time step dt

        Example:
        --------
        >>> from SystemParameters import MuscleParameters
        >>> import Muscle
        >>> muscle_parameters = MuscleParameters()
        >>> muscle1 = Muscle.Muscle(muscle_parameters)
        >>> muscle1.stim = 0.05
        >>> muscle1.deltaLength = 0.01
        >>> muscle1.step(dt)
        """

        # Muscle specific parameters initialization
        self.l_slack = parameters.l_slack
        self.l_opt = parameters.l_opt
        self.v_max = parameters.v_max
        self.F_max = parameters.f_max
        self.pennation = parameters.pennation
        self.E_ref = 0.04  # Reference strain
        self.w = 0.4

        # Muscle parameters initialization
        self.musclejoints = []
        self.l_SE = 0.0  # Muscle Series Element Length
        self.l_CE = 0.0  # Muscle Contracticle Element Length
        self.A = 0.01  # Muscle activation
        self.stim = 0.01  # base stimulation
        self.l_MTC = 0.0  # Muscle Tendon Unit (MTU) length
        self.deltaLength = 0.0

        self.initializeMuscleLength()

    #########################  Attributes #########################
    @property
    def l_slack(self):
        """Muscle Tendon Slack Length."""
        return self.__l_slack

    @l_slack.setter
    def l_slack(self, value):
        """ Keyword Arguments:
        value -- "Muscle Tendon Slack Length. """
        self.__l_slack = value

    @property
    def l_MTC(self):
        """ Length of Muscle Tendon Complex."""
        return self.__l_MTC

    @l_MTC.setter
    def l_MTC(self, value):
        """ Keyword Arguments:
        value -- Length of Muscle Tendon Complex """
        self.__l_MTC = value

    @property
    def l_CE(self):
        """ Length of muscle contracticle element."""
        return self.__l_CE

    @l_CE.setter
    def l_CE(self, value):
        """ Keyword Arguments:
        value --  Length of muscle contracticle element"""
        self.__l_CE = value

    @property
    def activeForce(self):
        """This function returns the active force generated by the muscle."""
        return self.computeMuscleActiveForce(self.l_CE, self.v_CE, self.A)

    @property
    def passiveForce(self):
        """This function returns the passive force generated by the muscle."""
        return self._F_PE_star(self.l_CE) + self._F_BE(self.l_CE)

    @property
    def v_CE(self):
        """Velocity of muscle contracticle element"""
        return self.__v_CE

    @v_CE.setter
    def v_CE(self, value):
        """Velocity of muscle contracticle element."""
        self.__v_CE = value

    @property
    def deltaLength(self):
        """This function returns the change in length of the muscle"""
        return self._deltaLength

    @deltaLength.setter
    def deltaLength(self, value):
        """ Keyword Arguments:
            value -- Set the change in Muscle Tendon Complex length"""
        self._deltaLength = value

    @property
    def force(self):
        """Function returns the sum of active and passive force"""
        return self.activeForce + self.passiveForce

    @property
    def tendonForce(self):
        """This function returns the force generated by the muscle."""
        return self.computeMuscleTendonForce(self.l_CE)

    ######################### METHODS #########################

    def computeMuscleActiveForce(self, l_CE, v_CE, a):
        """This function computes the Active Muscle Force.
        The function requires
        l_CE : Contracticle element length
        v_CE : Contracticle element velocity
        a : muscle activation."""
        return a * self._f_v_ce(v_CE) * self._f_l(l_CE) * self.F_max

    def computeMuscleTendonForce(self, l_CE):
        """This function computes the muscle tendon force.
        The function requires contracticle element length"""
        return self._F_SE(self.computeTendonLength(l_CE))

    def computeTendonLength(self, l_CE):
        """This function computes the muscle tendon length.
        The function requires contracticle element length"""
        return self.l_MTC - l_CE

    def computeMuscleTendonLength(self):
        """This function computes the total muscle length of muscle tendon unit.
        The function requires the list of muscle joint objects"""
        # if(self.deltaLength > 0.5):
        #     print('Delta Length : {}'.format(self.deltaLength))
        self.l_MTC = self.l_slack + self.l_opt + self.deltaLength
        for link in self.musclejoints:
            self.l_MTC += self.pennation * link.getDelta_Length()

    def updateActivation(self, dt):
        """This function updates the activation function of the muscle.
        The function requires time step dt as the inputs"""
        self.stim = max(0.01, min(1.0, self.stim))
        self.dA = (self.stim - self.A) * dt / Muscle.tau_act
        self.A += self.dA

    def initializeMuscleLength(self):
        """This function initializes the muscle lengths."""
        self.computeMuscleTendonLength()

        if(self.l_MTC < (self.l_slack + self.l_opt)):
            self.l_CE = self.l_opt
            self.l_SE = self.l_MTC - self.l_CE
        else:
            if(self.l_opt * self.w + self.E_ref * self.l_slack != 0.0):
                self.l_SE = self.l_slack * ((self.l_opt * self.w + self.E_ref * (
                    self.l_MTC - self.l_opt)) / (self.l_opt * self.w + self.E_ref * self.l_slack))
            else:
                self.l_SE = self.l_slack

            self.l_CE = self.l_MTC - self.l_SE

    def step(self, dt):
        """This function integrates and steps the muscle model by
        time step dt."""
        self.updateActivation(dt)
        self.computeMuscleTendonLength()
        self.v_CE = self._v_CE(
            self._f_v(
                self._F_SE(
                    self.l_SE), self._F_BE(
                    self.l_CE), self.A, self._f_l(
                    self.l_CE), self._F_PE_star(
                        self.l_CE)))
        # Integration of the velocity to obtain the muscle length
        # Here a simple EULER integration is applied since the integration
        # time steps are small
        # Can be changed to a more complex integrator
        self.l_CE = self.l_CE - self.v_CE * dt if self.l_CE > 0.0 else 0.0
        self.l_SE = self.l_MTC - self.l_CE

    def applyForce(self):
        # Applies force to the Joint
        """This function applies the force to the respective muscle joint.
        The function requires the list of muscle joint objects"""
        for link in self.musclejoints:
            link.addTorqueToJoint()

    def _F_SE(self, l_SE):
        """This function computes the Force in the Series Element (SE).
        The function requires SE length l_SE as inputs."""
        f_se = self.F_max * ((l_SE - self.l_slack) / (self.l_slack *
                                                      self.E_ref))**2 if l_SE > self.l_slack else 0.0
        return f_se

    def _F_PE_star(self, l_CE):
        """ This function computes the Force in the Parallel Element (PE).
        Force prevents the muscle from over-exentsion
        The function requires contracticle length l_CE as inputs."""
        return self.F_max * \
            ((l_CE - self.l_opt) / (self.l_opt * self.w))**2 if l_CE > self.l_opt else 0.0

    def _F_BE(self, l_CE):
        """ This function computes the Force in the muscle belly.
        Force prevents the muscle from collapsing on itself.
        The function requires SE length l_SE as inputs."""
        return self.F_max * ((l_CE - self.l_opt * (1.0 - self.w)) / (
            self.l_opt * self.w / 2.0))**2 if l_CE <= self.l_opt * (1.0 - self.w) else 0.0

    def _f_l(self, l_CE):
        """ This function computes the force from force-length relationship.
        The function requires SE length l_SE as inputs."""
        val = abs((l_CE - self.l_opt) / (self.l_opt * self.w))
        exposant = Muscle.c * val * val * val
        return np.exp(exposant)

    def _f_v_ce(self, v_CE):
        """ This function computes the force from force-velocity relationship.
        The function requires contracticle velocity as inputs."""
        if(v_CE >= 0):
            return (self.v_max - v_CE) / (self.v_max + Muscle.K * v_CE)
        else:
            return Muscle.N + (Muscle.N - 1) * (self.v_max +
                                                v_CE) / (7.56 * Muscle.K * v_CE - self.v_max)

    def _f_v(self, F_SE, F_BE, a, f_l, F_PE_star):
        """ This function computes the force from force-velocity relationship.
        The function requires
        F_SE : Series element force
        F_BE : Muscle belly force
        a : muscle activation
        f_l : Force from force-length relationship
        F_PE_star : Parallel element force."""
        if(self.F_max * a * f_l + F_PE_star == 0.0):
            f_v = 0.0
        else:
            f_v = (F_SE + F_BE) / ((self.F_max * a * f_l) + F_PE_star)

        f_v = 1.5 if f_v > 1.5 else f_v
        f_v = 0.0 if f_v < 0.0 else f_v

        return f_v

    def _v_CE(self, f_v):
        """ This function computes the Contracticle element velocity.
        The function requires force from force-velocity relationship."""
        return self.v_max * (1.0 - f_v) / (1.0 + f_v * Muscle.K) if f_v < 1.0 else self.v_max * \
            (f_v - 1.0) / (7.56 * Muscle.K * (f_v - Muscle.N) + 1.0 - Muscle.N)

