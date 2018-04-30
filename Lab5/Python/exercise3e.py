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

def frequency_effect(sim):
    #Effect of frequencies
    stim_frequency_range = np.array([0.05,0.1,0.5,1,5,10,50,100,500]) #in Hz
#    stim_frequency_range = np.array([1]) #in Hz
    stim_amp = 1 # belongs to 0-1
    phase_shift = np.pi
    frequency_pend=np.zeros(len(stim_frequency_range))
    amplitude_pend=np.zeros(len(stim_frequency_range))
    
    for j,stim_frequency in enumerate(stim_frequency_range):
        period = 1/stim_frequency
        t_max = 5*period  # Maximum simulation time
        time_step = 0.001*period
        time = np.arange(0., t_max, time_step)  # Time vector

        act1 = np.zeros((len(time),1))
        act2 = np.zeros((len(time),1))
        act1[:,0] = stim_amp*(1 + np.sin(2*np.pi*stim_frequency*time))/2
        act2[:,0] = stim_amp*(1+ np.sin(2*np.pi*stim_frequency*time + phase_shift))/2
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()  
        #computing the freuquency and amplitude
        angular_position = res[:,1]
        frequency_pend[j] = find_frequency(angular_position,time_step, int(len(angular_position)/2))
        amplitude_pend[j] = find_amplitude(angular_position, int(len(angular_position)/2))
    
    plt.figure()
    plt.subplot(121)
    plt.loglog(stim_frequency_range,frequency_pend)
    plt.grid()
    plt.xlabel('Stimulation Frequency in Hz')
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.subplot(122)
    plt.loglog(stim_frequency_range,amplitude_pend)
    plt.grid()
    plt.xlabel('Stimulation Frequency in Hz')
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.savefig('3_e1.png')
    plt.show()


def amplitude_effect(sim):
    stim_frequency = 10 #in Hz
    stim_amp_range = np.arange(0,1.1,0.1)# belongs to 0-1
    phase_shift = np.pi
    frequency_pend=np.zeros(len(stim_amp_range))
    amplitude_pend=np.zeros(len(stim_amp_range))
    
    for j,stim_amp in enumerate(stim_amp_range):
        period = 1/stim_frequency
        t_max = 5*period  # Maximum simulation time
        time_step = 0.001*period
        time = np.arange(0., t_max, time_step)  # Time vector

        act1 = np.zeros((len(time),1))
        act2 = np.zeros((len(time),1))
        act1[:,0] = stim_amp*(1 + np.sin(2*np.pi*stim_frequency*time))/2
        act2[:,0] = stim_amp*(1+ np.sin(2*np.pi*stim_frequency*time + phase_shift))/2
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()  
        #computing the freuquency and amplitude
        angular_position = res[:,1]
        frequency_pend[j] = find_frequency(angular_position,time_step, int(len(angular_position)/2))
#        frequency_pend[j] = find_freq_fft(angular_position,time_step)
        amplitude_pend[j] = find_amplitude(angular_position, int(len(angular_position)/2))
    
    frequency_pend[0] = 0.0;
    plt.figure()
    plt.subplot(121)
    plt.plot(stim_amp_range,frequency_pend)
    plt.grid()
    plt.xlabel('Stimulation Amplitude')
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.subplot(122)
    plt.plot(stim_amp_range,amplitude_pend)
    plt.grid()
    plt.xlabel('Stimulation Amplitude')
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.savefig('3_e2.png')
    plt.show()

#def exercise3e()
if __name__ == '__main__':

    """ Main function for question 3e

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

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

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
    
    #Effect of frequencies
    frequency_effect(sim)
    
    #Effect of stim amplitudes
    amplitude_effect(sim)