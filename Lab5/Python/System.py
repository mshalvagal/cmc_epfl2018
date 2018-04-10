""" This file implements the pendulum system with two muscles attached """

from SystemParameters import PendulumParameters, MuscleParameters
from Muscle import Muscle
import numpy as np
import biolog
import pdb
from copy import deepcopy
from matplotlib import pyplot as plt
from biopack import integrate
from scipy.integrate import odeint
from PendulumSystem import Pendulum
from MuscleSystem import MuscleSytem


class System(object):
    """Wrapper class for defining systems to simulate.
    """

    def __init__(self):
        super(System, self).__init__()
        self.systems_list = []

    def add_pendulum_system(self, system):
        """Add the pendulum system.

        Parameters
        ----------
        system: Pendulum
            Pendulum system
        """
        if self.systems_list.count('pendulum') == 1:
            biolog.warning(
                'You have already added the pendulum model to the system.')
            return
        else:
            biolog.info('Added pedulum model to the system')
            self.systems_list.append('pendulum')
            self.pendulum_sys = system

    def add_muscle_system(self, system):
        """Add the muscle system.

        Parameters
        ----------
        system: MuscleSystem
            Muscle system
        """
        if self.systems_list.count('muscle') == 1:
            biolog.warning(
                'You have already added the muscle model to the system.')
            return
        else:
            biolog.info('Added muscle model to the system')
            self.systems_list.append('muscle')
            self.muscle_sys = system

    def add_neural_system(self, system):
        """Add the neural network system.

        Parameters
        ----------
        system: NeuralSystem
            Neural Network system
        """
        if self.systems_list.count('neural') == 1:
            biolog.warning(
                'You have already added the neural model to the system.')
            return
        else:
            biolog.info('Added neural network to the system')
            self.systems_list.append('neural')
            self.neural_sys = system

    def show_system_properties(self):
        """Function prints the system properties in the system.
        """

        for sys in self.systems_list:
            if (sys == 'pendulum'):
                biolog.info(self.pendulum_sys.parameters.showParameters())
            elif (sys == 'muscle'):
                biolog.info(
                    self.muscle_sys.Muscle1.parameters.showParameters())
                biolog.info(
                    self.muscle_sys.Muscle2.parameters.showParameters())
            elif (sys == 'neural'):
                biolog.info(self.neural_sys.parameters.showParameters())

