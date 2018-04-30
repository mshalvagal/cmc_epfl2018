#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:37:46 2018

@author: manu
"""

""" Lab 5 Exercise 3f

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

def find_frequency(signal,time_step,index_start):
    signal_stat = signal[index_start:len(signal)]
    index_zeros = np.where(np.diff(np.sign(signal_stat)))[0] #np.where(signal_stat==0)[0]
    deltas = np.diff(index_zeros)
    delta = np.mean(deltas)
    period = 2*delta*time_step
    return 1/period

def find_amplitude(signal,index_start):
    signal_stat = signal[index_start:len(signal)]
    amplitude = (np.max(signal_stat)-np.min(signal_stat))/2
    return amplitude

def exercise3f():
    """ Function to run for Exercise 3f.

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
    sys.add_muscle_system(muscles)  # Add the muscle model to the system
    sim = SystemSimulation(sys)

    plt.close('all')
    t_max = 2.
    time_step = 0.001
    time = np.arange(0., t_max, time_step)
    act1 = np.zeros((len(time),1))
    act2 = np.zeros((len(time),1))
    act1[:,0] = 1*(1 + np.sin(2*np.pi*10*time))/2
    act2[:,0] = 1*(1+ np.sin(2*np.pi*10*time + np.pi))/2
    activations = np.hstack((act1, act2))
    sim.add_muscle_activations(activations)
    
    x0_P = np.array([0.0, 0.0])
    x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])
    x0 = np.concatenate((x0_P, x0_M))
    
    m1_origin = np.array([-0.2, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 1
    m2_origin = np.array([0.2, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 2
    
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))
    
    muscle_forces = np.logspace(-2,4,20)#[1,5,10,100,1500]
    
    frequency_pend=np.zeros(len(muscle_forces))
    amplitude_pend=np.zeros(len(muscle_forces))
    
    for i,force in enumerate(muscle_forces):
       sys.muscle_sys.Muscle1.F_max = force
       sys.muscle_sys.Muscle2.F_max = force
       
       sim.initalize_system(x0, time)
       sim.simulate()
       res = sim.results()
       
       #computing the freuquency and amplitude
       angular_position = res[:,1]
       frequency_pend[i] = find_frequency(angular_position,time_step, int(len(angular_position)/2))
       amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
       
   
    plt.figure()
    plt.subplot(121)
    plt.loglog(muscle_forces,frequency_pend)
    plt.grid()
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.xlabel('Maximal Muscle Force [N]')
    plt.subplot(122)
    plt.loglog(muscle_forces,amplitude_pend)
    plt.grid()
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.xlabel('Maximal Muscle Force [N]')
    plt.savefig('3_f.png')
    plt.show()
   
    muscle_forces = [1,5,10,100,1500]
   
    frequency_pend=np.zeros(len(muscle_forces))
    amplitude_pend=np.zeros(len(muscle_forces))
   
    plt.figure('Theta')
    plt.title('Pendulum angle')
    plt.ylabel('Position [rad]')
    plt.xlabel('Time [s]')
   
    plt.figure('Pendulum 3e')
    plt.title('Pendulum Phase')
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad/s]')
   
    labels = []
   
    for i,force in enumerate(muscle_forces):
       sys.muscle_sys.Muscle1.F_max = force
       sys.muscle_sys.Muscle2.F_max = force
       
       sim.initalize_system(x0, time)
       sim.simulate()
       res = sim.results()
       
       #computing the freuquency and amplitude
       angular_position = res[:,1]
       frequency_pend[i] = find_frequency(angular_position,time_step, int(len(angular_position)/2))
       amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
       
       plt.figure('Theta')
       plt.plot(time,angular_position)
       plt.grid()
       
       plt.figure('Pendulum 3e')
       plt.plot(res[:, 1], res[:, 2])
       plt.grid()
       
       labels.append(r'$F_{max}=$ ' + str(force))
   
   
    plt.figure('Pendulum 3e')
    plt.legend(labels)
    plt.savefig('3_f2.png')
   
    plt.figure('Theta')
    plt.legend(labels)
    plt.savefig('3_f3.png')
    
#    Asymmetric Muscles
    
    muscle_forces = np.logspace(-2,4,20)#[1,5,10,100,1500]
    
    frequency_pend=np.zeros(len(muscle_forces))
    amplitude_pend=np.zeros(len(muscle_forces))
    
    for i,force in enumerate(muscle_forces):
        sys.muscle_sys.Muscle1.F_max = force
        
        sim.initalize_system(x0, time)
        sim.simulate()
        res = sim.results()
        
        #computing the freuquency and amplitude
        angular_position = res[:,1]
        frequency_pend[i] = find_frequency(angular_position,time_step, int(len(angular_position)/2))
        amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
        
    
    plt.figure()
    plt.subplot(121)
    plt.loglog(muscle_forces,frequency_pend)
    plt.grid()
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.xlabel('Maximal Muscle Force [N]')
    plt.subplot(122)
    plt.loglog(muscle_forces,amplitude_pend)
    plt.grid()
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.xlabel('Maximal Muscle Force [N]')
    plt.savefig('3_f4.png')
    plt.show()    
    
    plt.figure('Theta')
    plt.title('Pendulum angle')
    plt.ylabel('Position [rad]')
    plt.xlabel('Time [s]')
    
    plt.figure('Pendulum 3e')
    plt.title('Pendulum Phase')
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad/s]')
    
    muscle_forces = [10,100,1500,2500]
    
    labels = []
    
    for i,force in enumerate(muscle_forces):
        sys.muscle_sys.Muscle1.F_max = force
        
        sim.initalize_system(x0, time)
        sim.simulate()
        res = sim.results()
        
        #computing the freuquency and amplitude
        angular_position = res[:,1]
        frequency_pend[i] = find_frequency(angular_position,time_step, int(len(angular_position)/2))
        amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
        
        plt.figure('Theta')
        plt.plot(time,angular_position)
        plt.grid()
        
        plt.figure('Pendulum 3e')
        plt.plot(res[:, 1], res[:, 2])
        plt.grid()
        
        labels.append(r'$F_{max,1}=$ ' + str(force))
    
    
    plt.figure('Pendulum 3e')
    plt.legend(labels)
    plt.savefig('3_f5.png')
    
    plt.figure('Theta')
    plt.legend(labels)
    plt.savefig('3_f6.png')

    
if __name__ == '__main__':
    exercise3e()
