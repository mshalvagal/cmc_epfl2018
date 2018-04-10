import numpy as np
from biopack import integrate
import pdb
import sys


class SystemSimulation(object):
    """System Simulation

    """

    def __init__(self, sys):
        super(SystemSimulation, self).__init__()
        self.sys = sys
        self.muscle_activations = None
        self.ext_in = None

    def add_muscle_activations(self, act):
        """Function applies the array of muscle activations during time
        integration Keyword Arguments: -- act <2D array>

        """
        self.muscle_activations = act

    def add_external_inputs_to_network(self, ext_in=None):
        """Function to add the external inputs to the neural network

        Parameters
        ----------
        self: type
            description
        ext_in: np.ndarray
            External inputs to each neuron in the network.
            Range of the inputs is [0, 1]
            The array is np.ndarray containing external inputs to
            each neuron at time t
        """
        self.ext_in = ext_in

    def _get_current_external_input_to_network(self, time):
        """Function to get the current external input to network.

        Parameters
        ----------
        self: type
            description
        time: float
            Current simulation time

        Returns
        -------
        current_ext_in : np.array
            Current external input to each neuron at time t
        """

        if(self.ext_in is not None):
            index = np.argmin((self.time - time)**2)
            return np.array(self.ext_in[index, :])
        else:
            return np.zeros(4)

    def _get_current_muscle_activation(self, time, state):
        """Function to return the current muscle activation to be applied
        during integration.

        """
        if (self.sys.systems_list.count('neural') == 1):
            # Apply the activation function to the neuron state m
            neural_act = self.sys.neural_sys.n_act(state[6:])
            return np.array([neural_act[0], neural_act[1]])
        else:
            if (self.muscle_activations is not None):
                index = np.argmin((self.time - time)**2)
                return np.array(self.muscle_activations[index, :])
            else:
                return np.array([0.05, 0.05])

    def muscle_joint_interface(self, time, state):
        activations = self._get_current_muscle_activation(time, state)
        self.sys.muscle_sys.Muscle1.stim = activations[0]
        self.sys.muscle_sys.Muscle2.stim = activations[1]
        torque = self.sys.muscle_sys.compute_muscle_torque(state[0])
        return torque

    def initalize_system(self, x0, time, *args):
        """Initialize the system to start simulation.

        Parameters
        ----------
        x0: numpy.array
            Initial states of the models on the system
        time: numpy.array
            Time vector for the system to be integrated for
        args: tuple
            external args for the integrator

        """

        self.x0 = x0
        self.time = time
        self.args = args

        # Initialize muscle states
        init_muscle_lce = self.sys.muscle_sys.initialize_muscle_length(
            self.x0[0])

        self.x0[3] = init_muscle_lce[0]
        self.x0[5] = init_muscle_lce[1]

        # Validate the muscle attachment points
        valid = self.sys.muscle_sys.validate_muscle_attachment(
            self.sys.pendulum_sys.parameters)
        if(not valid):
            sys.exit(1)

    def derivative(self, state, time, *args):

        # Compute the joint torques from muscle forces
        torque = self.muscle_joint_interface(time, state)

        p_der = self.sys.pendulum_sys.derivative(
            state[:2], time, torque)

        m_der = self.sys.muscle_sys.derivative(
            state[2:6], time, state[0])

        if (self.sys.systems_list.count('neural') == 1.0):
            self.sys.neural_sys.external_inputs(
                self._get_current_external_input_to_network(time))
            n_der = self.sys.neural_sys.derivative(
                state[6:], time)
            update = np.concatenate((p_der, m_der, n_der))
        else:
            update = np.concatenate((p_der, m_der))

        return update

    def simulate(self):
        self.res = integrate(self.derivative, self.x0, self.time,
                             args=self.args, rk=True, tol=True)

    def results(self):
        """Return the state of the system after integration.

        The function adds the time vector to the integrated system states."""

        return np.concatenate(
            (np.expand_dims(self.time, axis=1), self.res), axis=1)

    def results_muscles(self):
        """Returns the internal states of the muscle after integration is completed.

        Parameters
        ----------
        None

        Returns
        -------
        res_muscles: dict
            Dictionary for Muscle 1 and Muscle 2.
            For MUSCLE 1
            res_muscles['muscle1']: np.ndarray
                res_muscles['muscle1'][0]: np.array
                    Contracticle element length[l_CE]
                res_muscles['muscle1'][1]: np.array
                    Contracticle element velocity[v_CE]
                res_muscles['muscle1'][2]: np.array
                    Length of muscle tendon unit[l_MTC]
                res_muscles['muscle1'][3]: np.array
                    Contracticle element force[activeForce]
                res_muscles['muscle1'][4]: np.array
                    Passive element force[passiveForce]
                res_muscles['muscle1'][5]: np.array
                    activeForce + passiveForce[force]
                res_muscles['muscle1'][6]: np.array
                    Muscle tendon Force[tendonForce]
            For MUSCLE 2
            res_muscles['muscle2']: np.ndarray
                res_muscles['muscle2'][0]: np.array
                    Contracticle element length[l_CE]
                res_muscles['muscle2'][1]: np.array
                    Contracticle element velocity[v_CE]
                res_muscles['muscle2'][2]: np.array
                    Length of muscle tendon unit[l_MTC]
                res_muscles['muscle2'][3]: np.array
                    Contracticle element force[activeForce]
                res_muscles['muscle2'][4]: np.array
                    Passive element force[passiveForce]
                res_muscles['muscle2'][5]: np.array
                    activeForce + passiveForce[force]
                res_muscles['muscle2'][6]: np.array
                    Muscle tendon Force[tendonForce]
        """

        angle = self.res[:, 0]

        m1_state = self.res[:, 2:4]
        m2_state = self.res[:, 4:6]

        # Initializing the muscles results dictionary
        res_muscles = {'muscle1': np.empty(
            (len(angle), 7)), 'muscle2': np.empty((len(angle), 7))}

        # Get the muscle objects
        m1 = self.sys.muscle_sys.Muscle1
        m2 = self.sys.muscle_sys.Muscle2

        # Iterate over the states to re compute the paramters
        for i, angle_ in enumerate(angle):

            delta_length = self.sys.muscle_sys.delta_length_from_angle(angle_)

            # Muscle 1
            res_muscles['muscle1'][i, :] = m1.ode_result(
                m1_state[i, 0], m1_state[i, 1], delta_length[0])
            # Muscle 2
            res_muscles['muscle2'][i, :] = m2.ode_result(
                m2_state[i, 0], m2_state[i, 1], delta_length[1])

        return res_muscles

