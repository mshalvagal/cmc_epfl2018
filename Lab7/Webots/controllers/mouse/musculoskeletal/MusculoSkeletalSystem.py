""" MusculoSkeletalSystem class """

import os
import sys
import json

import numpy as np

import Joint as biojoint
import Muscle as biomuscle
import MuscleJoint as biomusclejoint
from SystemParameters import MuscleParameters
from SystemParameters import JointParameters
from SystemParameters import MuscleJointParameters


class MusculoSkeletalSystem(object):
    """ MusculoSkeletalSystem """

    def __init__(self, config_file=None):
        """ Initialize the joints and muscles.
        Need to initialize the class with a valid json file"""
        if config_file is None:
            raise RuntimeError('Missing config file ..... \n\n\n')
        elif not os.path.isfile(config_file):
            raise RuntimeError('Wrong config file ..... \n\n\n')
        else:
            with open(config_file) as f:
                parameters = json.load(f)
                self.joint_parameters = parameters['joints']
                self.muscle_parameters = parameters['muscles']
        # Create the dictionaries to store simulated joints and muscles of
        # the biomech model
        self.sim_joints = {}
        self.sim_muscles = {}

        # Store muscle names and joint names
        self.sim_joint_names = []
        self.sim_muscle_names = []

        # Create the simulated joints
        self.create_joints()
        # Create the simulated muscles
        self.create_muscles()
        # Create the simulated muscle-joint interface
        self.create_muscle_joints()

        # Torque
        self.torque = {}

    def generate_muscle_parameters(self, muscle_name):
        """To create muscle parameters from json file.

        Parameters
        ----------
        muscle_name: string
            Name of the muscle
        """
        param = self.muscle_parameters[muscle_name]
        if param is None:
            print('Invalid muscle name')
            sys.exit(1)
        return MuscleParameters(
            l_slack=param['l_slack'],
            l_opt=param['l_opt'],
            v_max=param['v_max'],
            f_max=param['f_max'],
            pennation=param['pennation'],
            name=muscle_name)

    def generate_muscle_joint_parameters(self, muscle_name):
        """To create muscle-joint parameters from json file.

        Parameters
        ----------
        muscle_name: string
            Name of the muscle
        """

        param = self.muscle_parameters[muscle_name]
        if param is None:
            print('Invalid muscle name')
            sys.exit(1)

        if param['muscle_type'] == 'mono':
            return [MuscleJointParameters(
                muscle_type=param['muscle_type'],
                r_0=param['r_0'],
                joint_attach=param['joint_attach'],
                theta_max=param['theta_max'],
                theta_ref=param['theta_ref'],
                direction=param['direction'])]

        elif param['muscle_type'] == 'bi':
            return [
                MuscleJointParameters(
                    muscle_type=param['muscle_type'],
                    r_0=param['r_0'],
                    joint_attach=param['joint_attach'],
                    theta_max=param['theta_max'],
                    theta_ref=param['theta_ref'],
                    direction=param['direction']
                ),
                MuscleJointParameters(
                    muscle_type=param['muscle_type'],
                    r_0=param['r_02'],
                    joint_attach=param['joint_attach2'],
                    theta_max=param['theta_max2'],
                    theta_ref=param['theta_ref2'],
                    direction=param['direction2']
                )
            ]

    def generate_joint_parameters(self, joint_name):
        """To create joint parameters from json file.

        Parameters
        ----------
        joint_name: string
            Name of the joint
        """

        param = self.joint_parameters[joint_name]
        if param is None:
            print('Invalid joint name')
            sys.exit(1)
        return JointParameters(
            name=joint_name,
            theta_max=param['theta_max'],
            theta_min=param['theta_min'],
            joint_type=param['joint_type'],
            reference_angle=param['reference_angle'])

    def create_joints(self):
        """This function creates joint objects based on the config file.
        The function stores the created joint objects in a dict."""
        for joint in self.joint_parameters:
            self.sim_joints[joint] = biojoint.Joint(
                0.0, self.generate_joint_parameters(joint))
            print(
                'Created joint : {} of type {} '.format(
                    joint,
                    self.joint_parameters[joint]['joint_type']
                )
            )

    def create_muscles(self):
        """This function creates muscle objects based on the config file.
        The function stores the created muscle objects in a dict."""
        for muscle in self.muscle_parameters:
            self.sim_muscle_names.append(muscle)
            param = self.generate_muscle_parameters(muscle)
            self.sim_muscles[muscle] = biomuscle.Muscle(param)
            print('Created muscle : ' + self.sim_muscles[muscle].name)

    def create_muscle_joints(self):
        """ Creates the muscle joint interface."""
        for muscle_name in self.muscle_parameters:
            muscle_joint_param = self.generate_muscle_joint_parameters(
                muscle_name)
            # Get the muscle
            muscle = self.sim_muscles[muscle_name]
            for mj in muscle_joint_param:
                joint = self.sim_joints[mj.joint_attach]
                muscle.musclejoints.append(
                    biomusclejoint.MuscleJoint(muscle, joint, mj))
                print(
                    (
                        'Created muscle joint interface'
                        ' for muscle {} and joint {}'
                    ).format(
                        muscle.name, joint.name)
                )

    def update_joints(self, joint_positions, dt):
        """ Update the joint angles.

        Parameters
        ----------
        joint_positions: dict
            Dictionary of joint angles
        """
        for joint in self.sim_joints:
            self.sim_joints[joint].updateAngle(joint_positions[joint])
            self.sim_joints[joint].step(dt)

    def apply_stim(self, muscle_stim=None):
        """ Apply muscle stimulation.

        Parameters
        ----------
       muscle_stim: dict
            Dictionary of muscle activation for each muscle
        """
        if muscle_stim is None:
            for muscle in self.sim_muscles:
                self.sim_muscles[muscle].stim = 0.05
        else:
            for muscle in self.sim_muscles:
                self.sim_muscles[muscle].stim = muscle_stim[muscle]

    def update_muscles(self, dt):
        """ Update the state of muscles.

        Parameters
        ----------
        dt: float
            Step integration time
        """
        steps = np.arange(0, 10, 1)
        for muscle in self.sim_muscles:
            self.sim_muscles[muscle].applyForce()
            for _ in steps:
                self.sim_muscles[muscle].step(dt / 10)

    def joint_torque(self):
        """ Compute the joint torque."""
        for joint in self.sim_joints:
            self.torque[joint] = self.sim_joints[joint].getTorque() * 1e7
        return self.torque

    def update(self, dt, joint_positions, muscle_stim=None):
        """ Step the complete bio-mechanical system.

        Parameters
        ----------
        self: type
            description
        dt: float
            Integration time step
        joint_positions: dict
            Dictionary of joint angles
        muscle_stim: dict
            Dictionary of muscle activations
        """
        self.update_joints(joint_positions, dt)
        self.apply_stim(muscle_stim)
        self.update_muscles(dt)
        return self.joint_torque()

# ----------------------------------------------------------------------------
