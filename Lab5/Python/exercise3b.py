#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:59:24 2018

@author: manu
"""

""" Lab 5 Exercise 3b

This file implements the pendulum system with two muscles attached

"""

from SystemParameters import PendulumParameters, MuscleParameters
from Muscle import Muscle
import numpy as np
import biolog
from matplotlib import pyplot as plt
from biopack import DEFAULT
from biopack.plot import save_figure
from PendulumSystem import Pendulum
from MuscleSystem import MuscleSytem
from SystemSimulation import SystemSimulation
from SystemAnimation import SystemAnimation
from System import System

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def exercise3b():
    """ Function to run for Exercise 3b.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    # Define and Setup your pendulum model here
    # Check Pendulum.py for more details on Pendulum class
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    P_params.L = 0.5  # To change the default length of the pendulum
    P_params.mass = 1.  # To change the default mass of the pendulum
    pendulum = Pendulum(P_params)  # Instantiate Pendulum object

    #### CHECK OUT Pendulum.py to ADD PERTURBATIONS TO THE MODEL #####

    biolog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    biolog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sim = SystemSimulation(sys)

    plt.close('all')
    t_max = 1.
    time = np.arange(0., t_max, 0.001)
    act1 = np.zeros((len(time), 1))
    act2 = np.zeros((len(time), 1))
    activations = np.hstack((act1, act2))
    x0_P = np.array([np.pi/4, 0.0])
    
    attachment_points = [0.01,0.1,0.15]
    m1_origin = np.array([-0.2, 0.0])  # Origin of Muscle 1
    m2_origin = np.array([0.2, 0.0])  # Origin of Muscle 2
    
    plt.figure('Pendulum 3b')
    plt.title('Pendulum Phase')
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad/s]')
    
    fig, ax = plt.subplots(nrows=2,num='m1')
    plt.title('Muscle 1')
    plt.xlabel('Time [s]')
    ax[0].set_title('Contractile element length')
    ax[0].set_ylabel('Muscle length [m]')
#    ax[1].set_title('Muscle velocity')
    ax[1].set_title('Muscle passive force')
    ax[1].set_ylabel('Muscle passive force [N]')
    
    fig, ax1 = plt.subplots(nrows=2,num='m2')
    plt.title('Muscle 2')
    plt.xlabel('Time [s]')
    ax1[0].set_title('Contractile element length')
    ax1[0].set_ylabel('Muscle length [m]')
#    ax1[1].set_title('Muscle velocity')
    ax1[1].set_title('Muscle passive force')
    ax1[1].set_ylabel('Muscle passive force [N]')
    
    plt.figure('theta')
    plt.title('Pendulum angle')
    plt.ylabel('Position [rad]')
    plt.xlabel('Time [s]')
    
    labels = []
    
    
    for i,a in enumerate(attachment_points):
        m1_insertion = np.array([0.0, -a])
        m2_insertion = np.array([0.0, -a])
        muscles.attach(np.array([m1_origin, m1_insertion]),
                       np.array([m2_origin, m2_insertion]))
        
        sys = System()
        sys.add_pendulum_system(pendulum)
        sys.add_muscle_system(muscles)

        x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])
        x0 = np.concatenate((x0_P, x0_M))
        sim.sys = sys
        sim.add_muscle_activations(activations)
        
        sim.initalize_system(x0, time)
        sim.simulate()
        res = sim.results()
        res_muscles = sim.results_muscles()
        
        plt.figure('Pendulum 3b')
        plt.plot(res[:, 1], res[:, 2])
        plt.grid()
        
        plt.figure('theta')
        plt.plot(time, res[:, 1])
        plt.grid()
    
        ax[0].plot(time, res_muscles['muscle1'][:,0])
#        ax[1].plot(time, res_muscles['muscle1'][:,1])
        ax[1].plot(time, res_muscles['muscle1'][:,4])
        plt.grid()
    
        ax1[0].plot(time, res_muscles['muscle2'][:,0])
#        ax1[1].plot(time, res_muscles['muscle2'][:,1])
        ax1[1].plot(time, res_muscles['muscle2'][:,4])
        plt.grid()
        
        labels.append(r'$a_2=$ ' + str(a))
    
    labels.append(r'$l_{opt}$')
    plt.figure('Pendulum 3b')
    plt.legend(labels)
    
    plt.figure('m1')
    ax[0].plot(time,M1.l_opt*np.ones_like(time),linestyle='-.')
    ax[0].legend(labels)
    ax[1].legend(labels[0:-1])
    
    plt.figure('m2')
    ax1[0].plot(time,M2.l_opt*np.ones_like(time),linestyle='-.')
    ax1[0].legend(labels)
    ax1[1].legend(labels[0:-1])
        
    plt.figure('theta')
    plt.legend(labels[0:-1])
    
if __name__ == '__main__':
    exercise3b()

