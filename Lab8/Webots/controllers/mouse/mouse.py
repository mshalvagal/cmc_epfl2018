"""This file implements reflex controller."""
# Default modules
import os

# Webots modules
from controller import Supervisor
from controller import Keyboard

# MusculoSkeletal system
from musculoskeletal import MusculoSkeletalSystem

# Reflexes
from reflexes import Reflexes

# Muscle Visualization
from muscle_visualization import MuscleVisualization

import numpy as np

# VARIABLE TO TURN ON/OFF MUSCLE VISUALIZATION
MUSCLE_VISUALIZATION = False

# TRAJECTORIES BUFFER SIZE, CAN BE INCREASED FOR LONGER SIMULATIONS
BUFFER_SIZE_TRAJECTORIES = int(1e6)

# RESULTS DIRECTORY (INSIDE /Lab7/Webots/controllers/mouse/Results by default)
RESULTS_DIRECTORY = os.path.join(".", "Results", "")


class Mouse(Supervisor):
    """Main class for Mouse control. """

    def __init__(self):
        super(Mouse, self).__init__()
        self.biomech = MusculoSkeletalSystem.MusculoSkeletalSystem(
            os.path.join('musculoskeletal', 'mouse.json')
        )
        self.TIMESTEP = int(self.getBasicTimeStep())

        self.motors = {}
        self.position_sensors = {}
        self.ground_contact_sensors = {}
        self.joint_positions = {}
        self.muscle_forces = {}
        self.ground_contacts = {}

        # Initialize webots
        self.key_press = None
        self.initialize_webots_motors()
        self.initialize_webots_position_sensors()
        self.initialize_webots_ground_contact_sensors()
        self.initialize_webots_keyboard()

        # Muscle visualization
        self.gps = {}
        self.mv_transform = {}
        self.mv_color = {}
        self.mv_geom = {}
        self.muscle_viz = None
        self.initialize_gps()
        self.initialize_muscle_visualizations()

        # Time
        self.time = np.zeros([BUFFER_SIZE_TRAJECTORIES, 1])

        # Foot Trajectories [axis]
        self.ankle_r_trajectory = np.zeros([BUFFER_SIZE_TRAJECTORIES, 3])
        self.ankle_l_trajectory = np.zeros([BUFFER_SIZE_TRAJECTORIES, 3])

        # Ground contact
        self.foot_r_contact = np.zeros([BUFFER_SIZE_TRAJECTORIES, 1])
        self.foot_l_contact = np.zeros([BUFFER_SIZE_TRAJECTORIES, 1])

        # Muscle Activations and Forces
        self.muscle_lh_activations = np.zeros([BUFFER_SIZE_TRAJECTORIES, 8])
        self.muscle_rh_activations = np.zeros([BUFFER_SIZE_TRAJECTORIES, 8])
        self.muscle_lh_forces = np.zeros([BUFFER_SIZE_TRAJECTORIES, 8])
        self.muscle_rh_forces = np.zeros([BUFFER_SIZE_TRAJECTORIES, 8])

        # Joint Angles
        self.joint_lh_positions = np.zeros([BUFFER_SIZE_TRAJECTORIES, 4])
        self.joint_rh_positions = np.zeros([BUFFER_SIZE_TRAJECTORIES, 4])

        self.iteration = 0

    def run(self):
        """ Run """

        reflex = Reflexes(self.biomech.sim_muscle_names)

        # TO ENABLE/DISABLE REFLEX STATE AT THE BEGINNING OF SIMULATION
        # DO IT HERE
        # # reflex.COUPLING = True (IT IS ENABLED BY DEFAULT)

        while self.step(self.TIMESTEP) != -1:

            # TO ENABLE/DISABLE REFLEX STATE ANYTIME DURING THE SIMULATION
            # DO IT HERE

            # # if (self.getTime() == 1.):
            # #     reflex.COUPLING = False

            # Reflex model
            reflex.step(
                self.update_joint_positions(),
                self.update_muscle_forces(),
                self.update_ground_contacts()
            )

            activations = reflex.activations

            # Update the biomechanical model
            self.biomech.update(
                self.TIMESTEP / 1000.,
                self.update_joint_positions(),
                activations
            )

            # Get the Torque
            torque = self.biomech.joint_torque()

            for name, motor in self.motors.items():
                motor.setTorque(torque[name])

            if MUSCLE_VISUALIZATION:
                self.muscle_viz.step(activations, viz=True)
            else:
                self.muscle_viz.step(activations, viz=False)

            # Save the current time step data
            self.save_data()

    def initialize_webots_keyboard(self):
        """ Initialize webots keyboard """
        self.key_press = Keyboard()
        self.key_press.enable(self.TIMESTEP * 25)

    def initialize_webots_motors(self):
        """Set-up leg joints in the system."""
        for joint in self.biomech.sim_joints:
            print(('Initializing webots motor : {}'.format(joint)))
            self.motors[joint] = self.getMotor(str(joint))

    def initialize_webots_position_sensors(self):
        """Set-up leg joints in the system."""
        for joint in self.biomech.sim_joints:
            print(('Initializing webots motor : {}'.format(joint)))
            self.position_sensors[joint] = self.getPositionSensor(
                str(joint) + '_POS'
            )
            self.position_sensors[joint].enable(self.TIMESTEP)

    def initialize_webots_ground_contact_sensors(self):
        """Initialize groung contact sensors."""
        print('Initializing webots ground contact sensors ')
        # GET TOUCH SENSORS
        self.ground_contact_sensors['LEFT_TOE_TOUCH'] = self.getTouchSensor(
            'LEFT_TOE_TOUCH'
        )
        self.ground_contact_sensors['RIGHT_TOE_TOUCH'] = self.getTouchSensor(
            'RIGHT_TOE_TOUCH'
        )
        # ENABLE TOUCH SENSORS
        self.ground_contact_sensors['LEFT_TOE_TOUCH'].enable(self.TIMESTEP)
        self.ground_contact_sensors['RIGHT_TOE_TOUCH'].enable(self.TIMESTEP)

    # MUSCLE VISUALIZATION

    def initialize_muscle_visualizations(self):
        """Initialize necessary attributes for muscle visualization."""

        # Get muscle transform, appearance and geom
        for muscle in self.biomech.sim_muscle_names:
            muscle_split = muscle.split('_')
            side = muscle_split[0]
            name = muscle_split[-1]

            # TRANSFORM
            transform = str(side + '_MV_TRANSFORM_' + name)
            self.mv_transform[transform] = self.getFromDef(transform)
            # COLOR
            appearance = str(side + '_MV_COLOR_' + name)
            self.mv_color[appearance] = self.getFromDef(appearance)
            # GEOM
            geom = str(side + '_MV_GEOM_' + name)
            self.mv_geom[geom] = self.getFromDef(geom)
            # Creat muscle muscle visualization object
            self.muscle_viz = MuscleVisualization(
                self.muscle_visualization_attachment(),
                self.gps,
                self.mv_transform,
                self.mv_color,
                self.mv_geom)
        return

    def initialize_gps(self):
        """Initialize gps nodes for muscle visualization."""

        GPS_NAMES = ['G1_PELVIS', 'G2_PELVIS',
                     'G1_HIP', 'G2_HIP',
                     'G1_KNEE', 'G2_KNEE',
                     'G1_ANKLE']

        sides = ['LH', 'RH']

        for side in sides:
            for gps in GPS_NAMES:
                name = side + '_' + gps
                self.gps[name] = self.getGPS(name)
                self.gps[name].enable(self.TIMESTEP)
        return

    def muscle_visualization_attachment(self):
        """Returns the dictionaries muscle origin and
        insertion with respect to GPS."""

        muscle_attach = {
            'PMA': ['G1_PELVIS', 'G1_HIP'],
            'CF': ['G2_PELVIS', 'G1_HIP'],
            'SM': ['G2_PELVIS', 'G1_KNEE'],
            'POP': ['G1_HIP', 'G1_KNEE'],
            'RF': ['G1_HIP', 'G2_HIP'],
            'TA': ['G1_KNEE', 'G1_ANKLE'],
            'SOL': ['G1_KNEE', 'G2_KNEE'],
            'LG': ['G1_HIP', 'G2_KNEE']
        }
        return muscle_attach

    def update_joint_positions(self):
        """ Initialize the array to store joint positions."""
        for name, sensor in self.position_sensors.items():
            self.joint_positions[name] = (
                sensor.getValue()
                + self.biomech.sim_joints[name].reference_angle
            )
        return self.joint_positions

    def update_muscle_forces(self):
        """ Initialize the array to store joint positions."""
        for muscle in self.biomech.sim_muscles:
            self.muscle_forces[muscle] = self.biomech.sim_muscles[
                muscle
            ].tendonForce
        return self.muscle_forces

    def update_ground_contacts(self):
        """ Update ground contacts """
        self.ground_contacts['L'] = self.ground_contact_sensors[
            'LEFT_TOE_TOUCH'
        ].getValue()
        self.ground_contacts['R'] = self.ground_contact_sensors[
            'RIGHT_TOE_TOUCH'
        ].getValue()
        return self.ground_contacts

    def save_data(self):
        """ Save data from the simulation. """
        # SAVA DATA
        self.iteration += 1

        # Time
        self.time[self.iteration] = (
            self.getTime()
        )

        # Save data Foot Trajectories
        self.ankle_l_trajectory[self.iteration, :] = (
            self.gps["LH_G1_ANKLE"].getValues()
        )
        self.ankle_r_trajectory[self.iteration, :] = (
            self.gps["RH_G1_ANKLE"].getValues()
        )

        # Save Ground Contact Time
        self.foot_l_contact[self.iteration] = self.ground_contact_sensors[
            'LEFT_TOE_TOUCH'].getValue()
        self.foot_r_contact[self.iteration] = self.ground_contact_sensors[
            'RIGHT_TOE_TOUCH'].getValue()

        # Save Muscle Data
        (muscle_lh_activations,
         muscle_rh_activations,
         muscle_lh_forces,
         muscle_rh_forces) = self.biomech.results_muscles()

        self.muscle_lh_activations[self.iteration, :] = (
            muscle_lh_activations
        )
        self.muscle_rh_activations[self.iteration, :] = (
            muscle_rh_activations
        )
        self.muscle_lh_forces[self.iteration, :] = (
            muscle_lh_forces
        )
        self.muscle_rh_forces[self.iteration, :] = (
            muscle_rh_forces
        )

        # Save joint positions
        joint_lh_positions, joint_rh_positions = self.biomech.results_joints()
        self.joint_lh_positions[self.iteration, :] = (
            joint_lh_positions
        )
        self.joint_rh_positions[self.iteration, :] = (
            joint_rh_positions
        )

    def __del__(self):
        """ Deletion """
        try:
            os.stat(RESULTS_DIRECTORY)
        except:
            os.mkdir(RESULTS_DIRECTORY)
        np.save(
            RESULTS_DIRECTORY + "time.npy",
            self.time[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "ankle_l_trajectory.npy",
            self.ankle_l_trajectory[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "ankle_r_trajectory.npy",
            self.ankle_r_trajectory[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "foot_r_contact.npy",
            self.foot_r_contact[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "foot_l_contact.npy",
            self.foot_l_contact[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "muscle_lh_activations.npy",
            self.muscle_lh_activations[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "muscle_rh_activations.npy",
            self.muscle_rh_activations[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "muscle_lh_forces.npy",
            self.muscle_lh_forces[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "muscle_rh_forces.npy",
            self.muscle_rh_forces[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "joint_lh_positions.npy",
            self.joint_lh_positions[:self.iteration, :]
        )
        np.save(
            RESULTS_DIRECTORY + "joint_rh_positions.npy",
            self.joint_rh_positions[:self.iteration, :]
        )
        return


def main():
    """ Main """
    mouse = Mouse()
    mouse.run()


if __name__ == '__main__':
    main()
