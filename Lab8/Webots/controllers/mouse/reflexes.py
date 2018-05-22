"""This file implements the reflexes based on the rules defined by Ekeberg."""

import numpy as np


class Reflexes(object):
    """ Class to describe the reflexes for locomotion. """

    def __init__(self, muscle_names):
        super(Reflexes, self).__init__()

        # Internal parameters
        self._COUPLING = None
        self._HIP_EXTENSION_RULE = None
        self._ANKLE_UNLOADING_RULE = None

        # Attributes
        self.muscle_names = muscle_names
        self.activations = {}
        self.leg_curr_phase = {'L': 'TOUCH_DOWN', 'R': 'LIFT_OFF'}
        self.leg_prev_phase = {'L': 'SWING', 'R': 'STANCE'}
        self.side = None
        self.angles = None
        self.ground_contact = None
        self.forces = {}

        # REFLEX CONDITIONS TO BE STUDIED
        self.COUPLING = True
        self.HIP_EXTENSION_RULE = True
        self.ANKLE_UNLOADING_RULE = True

        # Initialize activation dictionary
        for name in self.muscle_names:
            print(name)
            self.activations[name] = 0.05

    def step(self, angles, muscle_forces, ground_contact):
        """ Step/Update the state of the reflexes
        Parameters
        ----------
        angles: dict
            Dictionary containing the joint angles
        muscle_forces: dict
            Dictionary containing muscle forces
        ground_contact: dict
            Dictionary containing bool values of foot contact
         """
        # Update angles
        self.angles = angles
        # Update muscle forces
        self.forces = muscle_forces
        # Update ground contact
        self.ground_contact = ground_contact

        # UNCOMMENT WHEN YOU HAVE TUNED ALL THE 4 PHASES
        # LEFT LEG
        self.state_transition('L')
        # RIGHT LEG
        self.state_transition('R')

        # # UNCOMMENT TO TUNE STANCE PHASE
        # self.stance_2_lift_off('L')
        # self.stance_2_lift_off('R')

        # # UNCOMMENT TO TUNE LIFT_OFF PHASE
        # self.lift_off_2_swing('L')
        # self.lift_off_2_swing('R')

        # # UNCOMMENT TO TUNE SWING PHASE
        # self.swing_2_touch_down('L')
        # self.swing_2_touch_down('R')

        # # UNCOMMENT TO TUNE TOUCH_DOWN PHASE
        # self.touch_down_2_stance('L')
        # self.touch_down_2_stance('R')

        return self.activations

    def contra_lateral_side(self, side):
        """Function returns the contra lateral side.

        Parameters
        ----------
        self: type
            description
        side: string
            String to compute contra lateral side of the leg
        """
        if side == 'L':
            return 'R'
        elif side == 'R':
            return 'L'

    def state_transition(self, side):
        """ Transition one state to another.
        1. Lift_off
        2. Swing
        3. TouchDown
        4. Stance

        Parameters
        ----------
        side: string
            String describing left or right hind limb
        """
        # PARAMETERS TO BE TUNED FOR UNLOADING/HIP ANGLE REFLEXES
        # EXECUTION OF LIFT_OFF PHASE
        HIP_ANGLE_LIFTOFF = -0.123
        ANKLE_UNLOADING_LIFTOFF = 0.25

        # PARAMETERS TO BE TUNED FOR HIP/KNEE ANGLE REFLEXES
        # EXECUTION OF TOUCH_DOWN PHASE
        HIP_ANGLE_TOUCHDOWN = 0.4
        KNEE_ANGLE_TOUCHDOWN = -1.25

        # COUPLING BETWEEN THE LEGS
        if self.COUPLING:
            # To make sure both legs do not take off at the same time
            contra_side = self.contra_lateral_side(side)
            contra_lateral_state = self.leg_curr_phase[contra_side] == 'STANCE'
        elif not self.COUPLING:
            contra_lateral_state = True

        # HIP EXTENSION RULE
        if self.HIP_EXTENSION_RULE:
            hip_extension_state = (
                self.angles[side + 'H_J_HIP'] < HIP_ANGLE_LIFTOFF)
        elif not self.HIP_EXTENSION_RULE:
            hip_extension_state = True

        # ANKLE UNLOADING RULE
        if self.ANKLE_UNLOADING_RULE:
            ankle_unloading_state = (
                self.forces[side + 'H_M_SOL'] < ANKLE_UNLOADING_LIFTOFF)
        elif not self.ANKLE_UNLOADING_RULE:
            ankle_unloading_state = True

        # EXECUTE STANCE PHASE AFTER GROUND CONTACT
        if (self.ground_contact[side]) and (
                self.leg_curr_phase[side] == 'TOUCH_DOWN') and (
                    self.leg_prev_phase[side] == 'SWING'):
            self.leg_curr_phase[side] = 'STANCE'
            self.leg_prev_phase[side] = 'TOUCH_DOWN'

        # EXECUTE LIFT_OFF PHASE WITH HIP ANGLE AND ANKLE UNLOADING REFLEXES
        # SET THE REFLEX CONDITIONS FOR HIP ANGLE AND ANKLE UNLOADING FORCE

        elif (hip_extension_state) and (
                ankle_unloading_state) and (
                    self.leg_curr_phase[side] == 'STANCE') and (
            self.leg_prev_phase[side] == 'TOUCH_DOWN') and(
                    contra_lateral_state):
            self.leg_curr_phase[side] = 'LIFT_OFF'
            self.leg_prev_phase[side] = 'STANCE'

        # EXECUTE SWING PHASE AFTER GROUND RELEASE

        elif (not self.ground_contact[side]) and (
                self.leg_curr_phase[side] == 'LIFT_OFF') and (
                    self.leg_prev_phase[side] == 'STANCE'):
            self.leg_curr_phase[side] = 'SWING'
            self.leg_prev_phase[side] = 'LIFT_OFF'

        # EXECUTE TOUCH_DOWN PHASE AFTER HIP AND KNEE ANGLE REFLEXES

        elif (self.angles[side + 'H_J_HIP'] > HIP_ANGLE_TOUCHDOWN) and (
                self.angles[side + 'H_J_KNEE'] < KNEE_ANGLE_TOUCHDOWN) and (
                self.leg_curr_phase[side] == 'SWING') and (
                self.leg_prev_phase[side] == 'LIFT_OFF'):
            self.leg_curr_phase[side] = 'TOUCH_DOWN'
            self.leg_prev_phase[side] = 'SWING'

        self.execute_state(side)
        return

    def execute_state(self, side):
        """Execute the transition from current state to next state."""
        if (self.leg_curr_phase[side] == 'STANCE') and (
                self.leg_prev_phase[side] == 'TOUCH_DOWN'
        ):
            self.stance_2_lift_off(side)

        elif (self.leg_curr_phase[side] == 'LIFT_OFF') and (
                self.leg_prev_phase[side] == 'STANCE'):
            self.lift_off_2_swing(side)

        elif (self.leg_curr_phase[side] == 'SWING') and (
                self.leg_prev_phase[side] == 'LIFT_OFF'):
            self.swing_2_touch_down(side)

        elif (self.leg_curr_phase[side] == 'TOUCH_DOWN') and (
                self.leg_prev_phase[side] == 'SWING'):
            self.touch_down_2_stance(side)
        return

    # Reflex states
    def stance_2_lift_off(self, side):
        """Transition from stance to lift-off.
        """
        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION FROM STANCE PHASE

        print(side + " STANCE")

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.6 #CF
        K2 = 0.9 #CF
        K3 = 0.7 #SM
        K4 = 0.6 #RF
        K5 = 0.7 #SOL
        K6 = 0.8 #LG

        self.activations[side + 'H_M_PMA'] = 0.05
        self.activations[side + 'H_M_CF'] = K1 if (
            self.angles[side + 'H_J_HIP'] > -0.349) else K2
        self.activations[side + 'H_M_SM'] = K3
        self.activations[side + 'H_M_POP'] = 0.01
        self.activations[side + 'H_M_RF'] = K4 + 0.2 * \
            self.forces[side + 'H_M_RF'] + 0.01 * \
            (self.angles[side + 'H_J_HIP'] - 0.8726)
        self.activations[side + 'H_M_TA'] = 0.01
        self.activations[side + 'H_M_SOL'] = K5 + \
            0.2 * self.forces[side + 'H_M_SOL']
        self.activations[side + 'H_M_LG'] = K6 * \
            0.5 * self.forces[side + 'H_M_TA']
        return

    def swing_2_touch_down(self, side):
        """Transition from swing to touch-down."""

        print(side + " SWING")

        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION TO SWING PHASE

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.5 #PMA
        K2 = 0.7 #POP
        K3 = 0.7 #TA
        
        self.activations[side + 'H_M_PMA'] = K1 + 0.01 * \
            (0.69813 - self.angles[side + 'H_J_KNEE'])
        self.activations[side + 'H_M_CF'] = 0.01
        self.activations[side + 'H_M_SM'] = 0.01
        self.activations[side + 'H_M_POP'] = K2
        self.activations[side + 'H_M_RF'] = 0.01
        self.activations[side + 'H_M_TA'] = K3 + 0.01 * \
            (0.69813 - self.angles[side + 'H_J_KNEE'])
        self.activations[side + 'H_M_SOL'] = 0.01
        self.activations[side + 'H_M_LG'] = 0.01
        return

    def touch_down_2_stance(self, side):
        """Transition from touch-down to stance."""

        print(side + " TOUCH_DOWN")

        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION TO TOUCH_DOWN PHASE

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.9 #CF
        K2 = 0.1 #SOL

        self.activations[side + 'H_M_PMA'] = 0.01
        self.activations[side + 'H_M_CF'] = K1
        self.activations[side + 'H_M_SM'] = 0.01
        self.activations[side + 'H_M_POP'] = 0.01
        self.activations[side + 'H_M_RF'] = 0.01
        self.activations[side + 'H_M_TA'] = 0.01
        self.activations[side + 'H_M_SOL'] = K2
        self.activations[side + 'H_M_LG'] = 0.01
        return

    def lift_off_2_swing(self, side):
        """Transition from lift-off to swing."""

        print(side + " LIFT_OFF")

        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION TO LIFT_OFF PHASE

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.6 #PMA
        K2 = 1.0 #TA

        self.activations[side + 'H_M_PMA'] = K1
        self.activations[side + 'H_M_CF'] = 0.01
        self.activations[side + 'H_M_SM'] = 0.05
        self.activations[side + 'H_M_POP'] = 0.01
        self.activations[side + 'H_M_RF'] = 0.01
        self.activations[side + 'H_M_TA'] = K2
        self.activations[side + 'H_M_SOL'] = 0.01
        self.activations[side + 'H_M_LG'] = 0.01
        return

    # PROPERTIES
    @property
    def COUPLING(self):
        """ Set the coupling between legs  """
        return self._COUPLING

    @COUPLING.setter
    def COUPLING(self, value):
        """
        Parameters
        ----------
        value: bool
            Set whether cross coupling between legs is active or not
        """
        print(('CHANGING COUPLING State to {}'.format(value)))
        self._COUPLING = value

    @property
    def HIP_EXTENSION_RULE(self):
        """ Return whether HIP EXTENSION RULE is active or not """
        return self._HIP_EXTENSION_RULE

    @HIP_EXTENSION_RULE.setter
    def HIP_EXTENSION_RULE(self, value):
        """
        Parameters
        ----------
        value: bool
            Set whether HIP EXTENSION RULE is active or not
        """
        print(('CHANGING HIP EXTENSION RULE state to {}'.format(value)))
        self._HIP_EXTENSION_RULE = value

    @property
    def ANKLE_UNLOADING_RULE(self):
        """ Return whether ANKLE UNLOADING RULE is active or not """
        return self._ANKLE_UNLOADING_RULE

    @ANKLE_UNLOADING_RULE.setter
    def ANKLE_UNLOADING_RULE(self, value):
        """
        Parameters
        ----------
        value: bool
            Set whether ANKLE UNLOADING RULE is active or not
        """
        print(('CHANGING ANKLE EXTENSION RULE state to {}'.format(value)))
        self._ANKLE_UNLOADING_RULE = value
