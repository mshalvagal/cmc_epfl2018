"""This file implements the reflexes based on the rules defined by Ekeberg."""


class Reflexes(object):
    """ Class to describe the reflexes for locomotion. """

    def __init__(self, muscle_names):
        super(Reflexes, self).__init__()
        # Attributes
        self.muscle_names = muscle_names
        self.activations = {}
        self.leg_curr_phase = {'L': 'TOUCH_DOWN', 'R': 'LIFT_OFF'}
        self.leg_prev_phase = {'L': 'SWING', 'R': 'STANCE'}
        self.cross_coupling = True
        self.side = None
        self.angles = None
        self.ground_contact = None
        self.forces = {}

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

        # # UNCOMMENT WHEN YOU HAVE TUNED ALL THE 4 PHASES
        # # LEFT LEG
        self.state_transition('L')
        # # RIGHT LEG
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

        # To make sure both legs do not take off at the same time
        contra_side = self.contra_lateral_side(side)

        # PARAMETERS TO BE TUNED FOR UNLOADING/HIP ANGLE REFLEXES
        # EXECUTION OF LIFT_OFF PHASE
        HIP_ANGLE_LIFTOFF = -0.1235
        ANKLE_UNLOADING_LIFTOFF = 0.8

        # PARAMETERS TO BE TUNED FOR HIP/KNEE ANGLE REFLEXES
        # EXECUTION OF TOUCH_DOWN PHASE
        HIP_ANGLE_TOUCHDOWN = 0.4
        ANKLE_UNLOADING_TOUCHDOWN = -1.25

        # EXECUTE STANCE PHASE AFTER GROUND CONTACT
        if (self.ground_contact[side]) and (
                self.leg_curr_phase[side] == 'TOUCH_DOWN') and (
                    self.leg_prev_phase[side] == 'SWING'):
            self.leg_curr_phase[side] = 'STANCE'
            self.leg_prev_phase[side] = 'TOUCH_DOWN'

        # EXECUTE LIFT_OFF PHASE WITH HIP ANGLE AND ANKLE UNLOADING REFLEXES
        # SET THE REFLEX CONDITIONS FOR HIP ANGLE AND ANKLE UNLOADING FORCE

        elif (self.angles[side + 'H_J_HIP'] < HIP_ANGLE_LIFTOFF) and (
                self.forces[side + 'H_M_SOL'] < ANKLE_UNLOADING_LIFTOFF) and (
                    self.leg_curr_phase[side] == 'STANCE') and (
                        self.leg_prev_phase[side] == 'TOUCH_DOWN') and (
                            self.leg_curr_phase[contra_side] == 'STANCE'):
            self.leg_curr_phase[side] = 'LIFT_OFF'
            self.leg_prev_phase[side] = 'STANCE'

        # EXECUTE SWING PHASE AFTER GROUND RELEASE

        elif (not self.ground_contact[side]) and (
                self.leg_curr_phase[side] == 'LIFT_OFF') and (
                    self.leg_prev_phase[side] == 'STANCE'):

            self.leg_curr_phase[side] = 'SWING'
            self.leg_prev_phase[side] = 'LIFT_OFF'

        # EXECUTE TOUCH_DOWN PHASE AFTER HIP AND KNEE ANGLE REFLEXES
        # SET THE REFLEX CONDITIONS FOR HIP ANGLE AND ANKLE UNLOADING FORCE

        elif (
                self.angles[side + 'H_J_HIP'] > HIP_ANGLE_TOUCHDOWN
        ) and (
            self.angles[side + 'H_J_KNEE'] < ANKLE_UNLOADING_TOUCHDOWN
        ) and (
            self.leg_curr_phase[side] == 'SWING'
        ) and (
            self.leg_prev_phase[side] == 'LIFT_OFF'
        ):
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

        # print("STANCE")

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.05
        K2 = 0.05
        K3 = 0.5
        K4 = 0.3
        K5 = 0.2
        K6 = 0.05

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

        # print("SWING")

        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION TO SWING PHASE

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.1
        K2 = 0.3
        K3 = 0.8

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

        # print("TOUCH_DOWN")

        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION TO TOUCH_DOWN PHASE

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.5
        K2 = 0.2

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

        # print("LIFT_OFF")

        # CHANGE THE ACTIVATION FUNCTION OF MUSCLES TO
        # TRANSITION TO LIFT_OFF PHASE

        # MUSCLE ACTIVATION CONSTANTS
        K1 = 0.3
        K2 = 0.7

        self.activations[side + 'H_M_PMA'] = K1
        self.activations[side + 'H_M_CF'] = 0.01
        self.activations[side + 'H_M_SM'] = 0.05
        self.activations[side + 'H_M_POP'] = 0.01
        self.activations[side + 'H_M_RF'] = 0.01
        self.activations[side + 'H_M_TA'] = K2
        self.activations[side + 'H_M_SOL'] = 0.01
        self.activations[side + 'H_M_LG'] = 0.01
        return
