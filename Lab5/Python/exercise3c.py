#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:54:16 2018

@author: matthieu
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
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels
plt.rcParams.update({'figure.autolayout': True}) # to get better graphs at save
plt.rcParams.update({'savefig.dpi': 500}) #set resolution for saving figures
plt.rcParams.update({'savefig.bbox': 'tight'}) #to include legends saving figures



#def exercise3c():
if __name__ == '__main__':
    """ Main function for question 3c

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
    
    #This pendulum will receive perturbations to prove limit cycle
    pendulum.perturb = True

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

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 4  # Maximum simulation time
    time_step = 0.001
    time = np.arange(0., t_max, time_step)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([0, 0])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent
    
    sin_frequency = 2 #in Hz
    amp_stim = 1
    phase_shift = np.pi
    act1 = np.zeros((len(time),1))
    act2 = np.zeros((len(time),1))
    for i in range(0,len(time)):
        act1[i,0] = amp_stim*(1+np.sin(2*np.pi*sin_frequency*time[i]))/2
        act2[i,0] = amp_stim*(1+ np.sin(2*np.pi*sin_frequency*time[i] + phase_shift))/2
    
    plt.figure()
    plt.plot(time,act1)
    plt.plot(time,act2)
    plt.legend(["Activation for muscle 1", "Activation for muscle 2"])
    plt.xlabel("Time [s]")
    plt.ylabel("Activation")
    plt.show()

    activations = np.hstack((act1, act2))

    # Method to add the muscle activations to the simulation

    sim.add_muscle_activations(activations)

    # Simulate the system for given time

    sim.initalize_system(x0, time)  # Initialize the system state

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res_1 = sim.results()

    # Plotting the results
    plt.figure('Pendulum')
    plt.title('Pendulum Phase')
    plt.plot(res_1[:, 1], res_1[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad/s]')
    plt.grid()
    plt.savefig('3_c.png')
    plt.show()
    
    # Plotting the results
    plt.figure()
    plt.title('Pendulum Oscillations')
    plt.plot(time,res_1[:, 1])
    plt.xlabel('Time [s]')
    plt.ylabel('Position [rad]')
    plt.grid()
    plt.show()
    