import numpy as np
import math
import biolog
import pdb


class MuscleSytem(object):
    """Class comprising of the antagonist muscle pair

    """

    def __init__(self, Muscle1, Muscle2):
        super(MuscleSytem, self).__init__()
        self.Muscle1 = Muscle1
        self.Muscle2 = Muscle2

        self.muscle1_length = self.Muscle1.l_MTC
        self.muscle2_length = self.Muscle2.l_MTC

    def attach(self, muscle1_pos, muscle2_pos):
        """ Muscle attachment points.

        Parameters:
        -----------
            muscle1_pos : <numpy.array>
                Attachment point of muscle 1.
                [origin,
                 insertion]
            muscle2_pos : <numpy.array>
                Attachment point of muscle 2.
                [origin,
                 insertion]

        Example:
        --------
        >>> muscle1_pos = numpy.array([[-5.0, 0.0],
                                       [0.0, 1.0]])
        >>> muscle2_pos = numpy.array([[5.0, 0.0],
                                       [0.0, 1.0]])
        """
        # Muscle 1 acts in negative torque direction
        self.dir1 = -1.0
        # Muscle 2 acts in positive torque direction
        self.dir2 = 1.0

        self.muscle1_pos = muscle1_pos
        self.muscle2_pos = muscle2_pos

        self.compute_attachment_distances()

    def compute_attachment_distances(self):
        """ Compute the distances between the pendulum joint
            and muscle attachment origin and inertion point.
        """

        # Muscle 1
        self.a1_m1 = np.linalg.norm(
            self.muscle1_pos[0] - np.array([0, 0]))
        self.a2_m1 = np.linalg.norm(
            self.muscle1_pos[1] - np.array([0, 0]))

        # Muscle 2
        self.a1_m2 = np.linalg.norm(
            self.muscle2_pos[0] - np.array([0, 0]))
        self.a2_m2 = np.linalg.norm(
            self.muscle2_pos[1] - np.array([0, 0]))

    def initialize_muscle_length(self, angle):
        """Initialize the muscle contractile and tendon length.

        Parameters
        ----------
        self: type
            description
        angle: float
            Initial position of the pendulum [rad]

        """
        delta_length = self.delta_length_from_angle(angle)
        self.Muscle1.deltaLength = delta_length[0]
        self.Muscle2.deltaLength = delta_length[1]

        self.Muscle1.initializeMuscleLength()
        self.Muscle2.initializeMuscleLength()
        return np.array([self.Muscle1.l_CE, self.Muscle2.l_CE])

    def validate_muscle_attachment(self, parameters):
        """Validate the muscle attachment positions.

        Provided pendulum length and muscle attachments
        check if the muscle attachments are valid or not!

        Parameters
        ----------
        parameters: PendulumParameters
            Pendulum parameters

        Returns
        -------
        check : bool
            Returns if the muscle attachments are valid or not
        """

        check = (parameters.L > abs(self.muscle1_pos[1, 1])) and (
            parameters.L > abs(self.muscle2_pos[1, 1])) and (
                self.muscle1_pos[1, 0] == 0.0) and (
                    self.muscle2_pos[1, 0] == 0.0)

        if check:
            biolog.info('Validated muscle attachments')
            return check
        else:
            biolog.error('Invalid muscle attachment points')
            return check

    def position_from_angle(self, angle):
        """ Compute the muscle position from joint angle.

        Parameters:
        -----------
            angle : <float>
                Pendulum angle

        Returns:
        --------
            muscle1_pos : <float>
                Updates Attachment points of muscle 1
            muscle2_pos : <float>
                Updates Attachment points of muscle 2
        """

        def rotMatrix(x): return np.array(
            [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])

        # Update muscle attachment points on the pendulum
        pos1 = np.array([self.muscle1_pos[0, :], np.dot(
            rotMatrix(angle), self.muscle1_pos[1, :])])

        pos2 = np.array([self.muscle2_pos[0, :], np.dot(
            rotMatrix(angle), self.muscle1_pos[1, :])])

        return np.concatenate(([pos1], [pos2]))

    def length_from_angle(self, angle):
        """ Compute the muscle length from joint angle.

        Parameters:
        -----------
            angle : <float>
                Pendulum angle

        Returns:
        --------
            muscle1_length : <float>
                Muscle 1 length
            muscle2_length : <float>
                Muscle 2 length
        """

        # Compute new muscle length
        self.muscle1_length = math.sqrt(
            self.a1_m1**2 + self.a2_m1**2 +
            2 * self.a1_m1 * self.a2_m1 * np.sin(angle))
        self.muscle2_length = math.sqrt(
            self.a1_m2**2 + self.a2_m2**2 -
            2 * self.a1_m2 * self.a2_m2 * np.sin(angle))
        return np.array([self.muscle1_length,
                         self.muscle2_length])

    def delta_length_from_angle(self, angle):
        """ Compute the muscle length from joint angle.

        Parameters:
        -----------
            angle : <float>
                Pendulum angle

        Returns:
        --------
            muscle1_delta_length : <float>
                Change in Muscle 1 length
            muscle2_delta_length : <float>
                Change in Muscle 2 length
        """

        return self.length_from_angle(angle) - np.array(
            [self.Muscle1.l_opt + self.Muscle1.l_slack,
             self.Muscle1.l_opt + self.Muscle1.l_slack])

    def compute_moment_arm(self, angle):
        """ Compute the moment arm of the muscles based on the joint angle.

        moment = a1*a2

        Parameters
        ----------
        angle: float
            Current angle of the pendulum

        """
        # Muscle 1 moment arm
        self.moment1 = self.a1_m1 * self.a2_m1 * \
            np.cos(angle) / self.muscle1_length
        # Muscle 2 moment arm
        self.moment2 = self.a1_m2 * self.a2_m2 * \
            np.cos(angle) / self.muscle2_length

    def compute_muscle_torque(self, angle):
        """ Keyword Arguments:
            angle -- Angle of the Pendulum """
        self.compute_moment_arm(angle)
        return self.dir1 * self.moment1 * self.Muscle1.tendonForce + \
            self.dir2 * self.moment2 * self.Muscle2.tendonForce

    def derivative(self, state, time, *args):
        """ Keyword Arguments:
            self  --
            state --
            time  --
            *args --  """

        # Set the change in muscle length

        angle = args[0]

        delta_length = self.delta_length_from_angle(angle)

        self.Muscle1.deltaLength = delta_length[0]
        self.Muscle2.deltaLength = delta_length[1]

        # Update and retrieve the derivatives
        return np.concatenate(
            (self.Muscle1.dydt(state[: 2]), self.Muscle2.dydt(state[2:])))

