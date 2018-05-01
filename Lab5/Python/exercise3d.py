#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:04:36 2018

@author: matthieu
"""

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



def find_freq_fft(signal,time_step):
    signal = signal-np.mean(signal)
    spectrum = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(spectrum))
    freq = freq/time_step
    index_max_freq = np.abs(spectrum) == np.max(np.abs(spectrum))
    freq_maxs = freq[index_max_freq]
    if len(freq_maxs)>2:
        print("Problem with finding frequency")  
    max_freq = np.abs(freq_maxs[0])
    return max_freq
    
def show_fft(signal,time_step):
    spectrum = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(spectrum))
    freq = freq/time_step
    plt.figure()
    plt.plot(freq, abs(spectrum))
    plt.xlim([0,10])
    plt.show()

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

def mass_effect(muscles,pendulum,act1,act2,x0,time,time_step,mass_range):
    frequency_pend=np.zeros(len(mass_range))
    amplitude_pend=np.zeros(len(mass_range))
    for i,mass in enumerate(mass_range):
        pendulum.parameters.mass = mass
        sys = System()  # Instantiate a new system
        sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
        sys.add_muscle_system(muscles)  # Add the muscle model to the system
        sim = SystemSimulation(sys)  # Instantiate Simulation object
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()
        angular_position = res[:,1]
        #frequency_pend[i] = find_frequency(angular_position,time_step)
        frequency_pend[i] = find_frequency(angular_position,time_step,int(len(angular_position)/2))
        amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
    plt.figure()
    plt.subplot(121)
    plt.semilogx(mass_range,frequency_pend)
    plt.xlabel('Mass of the pendulum in kg')
    plt.ylabel('Pendulum Oscillation Frequency in Hz')
    plt.subplot(122)
    plt.semilogx(mass_range,amplitude_pend)
    plt.xlabel('Mass of the pendulum in kg')
    plt.ylabel('Pendulum Oscillation Amplitude in rad')
    plt.savefig('3_d1.png')
    plt.show()
    
def length_effect(muscles,pendulum,act1,act2,x0,time,time_step,length_range):
    frequency_pend=np.zeros(len(length_range))
    amplitude_pend=np.zeros(len(length_range))
    for i,length in enumerate(length_range):
        pendulum.parameters.L = length
        sys = System()  # Instantiate a new system
        sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
        sys.add_muscle_system(muscles)  # Add the muscle model to the system
        sim = SystemSimulation(sys)  # Instantiate Simulation object
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()
        angular_position = res[:,1]
        frequency_pend[i] = find_frequency(angular_position,time_step,int(len(angular_position)/2))
        amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
    plt.figure()
    plt.subplot(121)
    plt.semilogx(length_range,frequency_pend)
    plt.xlabel('Length of the pendulum in m')
    plt.ylabel('Pendulum Oscillation Frequency in Hz')
    plt.subplot(122)
    plt.semilogx(length_range,amplitude_pend)
    plt.xlabel('Length of the pendulum in m')
    plt.ylabel('Pendulum Oscillation Amplitude in rad')
    plt.savefig('3_d2.png')
    plt.show()
    
def inertia_effect(muscles,pendulum,act1,act2,x0,time,time_step,inertia_range):
    frequency_pend=np.zeros(len(inertia_range))
    amplitude_pend=np.zeros(len(inertia_range))
    for i,inertia in enumerate(inertia_range):
        print("Inertia targeted {}".format(inertia))
        length = pendulum.parameters.L 
        pendulum.parameters.mass = 3*inertia/(length**2)
        #factor 3 in inertia, dont know why
        print("Inertia real {}".format(pendulum.parameters.I))
        sys = System()  
        sys.add_pendulum_system(pendulum) 
        sys.add_muscle_system(muscles)  
        sim = SystemSimulation(sys)  
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)
        sim.simulate()
        res = sim.results()
        angular_position = res[:,1]
        frequency_pend[i] = find_frequency(angular_position,time_step,int(len(angular_position)/2))
        amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
    plt.figure()
    plt.subplot(121)
    plt.semilogx(inertia_range,frequency_pend)
    plt.xlabel('Inertia of the pendulum in kg.m**2')
    plt.ylabel('Pendulum Oscillation Frequency in Hz')
    plt.subplot(122)
    plt.semilogx(inertia_range,amplitude_pend)
    plt.xlabel('Inertia of the pendulum in kg.m**2')
    plt.ylabel('Pendulum Oscillation Amplitude in rad')
    plt.savefig('3_d3.png')
    plt.show()
    
#does not work yet
def length_effect_constant_muscle_distance(muscles,pendulum,act1,act2,time,time_step,length_range,dist):
    frequency_pend=np.zeros(len(length_range))
    amplitude_pend=np.zeros(len(length_range))
    for i,length in enumerate(length_range):
        pendulum.parameters.L = length
        
        m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
        m1_insertion = np.array([0.0, -dist*length])  # Insertion of Muscle 1
        m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
        m2_insertion = np.array([0.0, -dist*length])  # Insertion of Muscle 2
        muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))
        
        x0_P = np.array([0, 0])  # Pendulum initial condition
        x0_M = np.array([0., muscles.Muscle1.l_CE, 0., muscles.Muscle2.l_CE])
        x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

        sys = System()  # Instantiate a new system
        sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
        sys.add_muscle_system(muscles)  # Add the muscle model to the system
        sim = SystemSimulation(sys)  # Instantiate Simulation object
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()
        angular_position = res[:,1]
        frequency_pend[i] = find_frequency(angular_position,time_step,int(len(angular_position)/2))
        amplitude_pend[i] = find_amplitude(angular_position, int(len(angular_position)/2))
        plt.figure()
        plt.plot(angular_position)
        plt.show()
    plt.figure()
    plt.subplot(121)
    plt.semilogx(length_range,frequency_pend)
    plt.xlabel('Length of the pendulum in m')
    plt.ylabel('Pendulum Oscillation Frequency in Hz')
    plt.subplot(122)
    plt.semilogx(length_range,amplitude_pend)
    plt.xlabel('Length of the pendulum in m')
    plt.ylabel('Pendulum Oscillation Amplitude in rad')
    #plt.savefig('3_d4.png')
    plt.show()

def exercise3d():

    """ Main function for question 3d

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

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))
    
    
    stim_frequency = 10 #in Hz
    stim_amp = 1 # between 0 and 1
    phase_shift = np.pi
    t_max = 5  # Maximum simulation time
    time_step = 0.001
    time = np.arange(0., t_max, time_step)  # Time vector
    act1 = np.zeros((len(time),1))
    act2 = np.zeros((len(time),1))
    for i in range(0,len(time)):
            act1[i,0] = stim_amp*(1 + np.sin(2*np.pi*stim_frequency*time[i]))/2
            act2[i,0] = stim_amp*(1+ np.sin(2*np.pi*stim_frequency*time[i] + phase_shift))/2
    
    plt.figure()
    plt.plot(time,act1)
    plt.plot(time,act2)
    plt.legend(["Activation for muscle 1", "Activation for muscle 2"])
    plt.xlabel("Time in s")
    plt.ylabel("Activation")
    plt.show()
    
    x0_P = np.array([0,0])
    x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])
    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions
    
    #Effect of mass
    #mass_range = np.array([0.01,0.03,0.1,0.5,1,5,10,30,100]) #in kg,default mass at 1 kg
    mass_range = np.array([30,100,300,1000]) #in kg,default mass at 1 kg
#    mass_effect(muscles,pendulum,act1,act2,x0,time,time_step,mass_range)
    
    #we reinitialize the pendulum
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum = Pendulum(P_params)  # Instantiate Pendulum object
  
    #Effect of length
    length_range = np.array([0.21,0.3,0.5,1,2,5,10,50]) #in m, default length at 0.5 m
#    length_effect(muscles,pendulum,act1,act2,x0,time,time_step,length_range)

    #we reinitialize the pendulum
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    #pendulum = Pendulum(P_params)  # Instantiate Pendulum object
    
    #Effect of inertia
    inertia_range = np.array([0.01,0.03,0.33,1,10,100]) #in kg.m**2, default inertia at 0.33 kg.m**2
    inertia_effect(muscles,pendulum,act1,act2,x0,time,time_step,inertia_range)
        
    
    #we reinitialize the pendulum
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum = Pendulum(P_params)  # Instantiate Pendulum object
    length_range = np.array([0.05,0.1,0.5,1,5]) #in m, default length at 0.5 m
    dist = 0.3 # between 0 and 1, muscle will be attached at dist*length
    #length_effect_constant_muscle_distance(muscles,pendulum,act1,act2,time,time_step,length_range,dist)
    

if __name__ == '__main__':
    exercise3d()

