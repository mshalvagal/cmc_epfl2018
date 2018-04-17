# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:19:58 2018

@author: Alexandre
"""

""" Lab 5 Exercise 3

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
    """ Main function to run for Exercise 3.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    plt.close('all')    
    #Question 3b   
   
    ##### Time #####
    t_max = 1.  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    #x0_P = np.array([0, 0],[np.pi/8, 0],[np.pi/4, 0],[np.pi/2, 0])  # Pendulum initial condition
    x0_P = np.array([np.pi/4, 0])
    
    
    # Define and Setup your pendulum model here
    # Check Pendulum.py for more details on Pendulum class
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    P_params.L = 0.5  # To change the default length of the pendulum
    P_params.mass = 1.  # To change the default mass of the pendulum
    pendulum = Pendulum(P_params)  # Instantiate Pendulum object
    
    #### CHECK OUT Pendulum.py to ADD PERTURBATIONS TO THE MODEL #####
    
    #biolog.info('Pendulum model initialized \n {}'.format(pendulum.parameters.showParameters()))
    
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
    #biolog.info('Muscle system initialized \n {} \n {}'.format(M1.parameters.showParameters(), M2.parameters.showParameters()))        
       
    #Symmetric Attachment
    a=np.linspace(0.01,0.3,4)
    leg=[]
    
    plt.figure('3b_1')
    plt.title('Pendulum Phase')
    for i in range(0,len(a)):
        leg.append('Origins= 0.2m ; Attachments= {}m'.format(a[i]))
        
        # Define Muscle Attachment points
        m1_origin = np.array([-0.2, 0.0])  # Origin of Muscle 1
        m1_insertion = np.array([0.0, -a[i]])  # Insertion of Muscle 1
    
        m2_origin = np.array([0.2, 0.0])  # Origin of Muscle 2
        m2_insertion = np.array([0.0, -a[i]])  # Insertion of Muscle 2
        
        # Attach the muscles
        muscles.attach(np.array([m1_origin, m1_insertion]),
                       np.array([m2_origin, m2_insertion]))
        
        # Muscle Model initial condition
        x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])

        x0 = np.concatenate((x0_P, x0_M))  # System initial conditions
    
        # Create a system with Pendulum and Muscles using the System Class
        # Check System.py for more details on System class
        sys = System()  # Instantiate a new system
        sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
        sys.add_muscle_system(muscles)  # Add the muscle model to the system
    
    
        ##### System Simulation #####
        # For more details on System Simulation check SystemSimulation.py
        # SystemSimulation is used to initialize the system and integrate
        # over time
    
        sim = SystemSimulation(sys)  # Instantiate Simulation object
    
        # Add muscle activations to the simulation
        # Here you can define your muscle activation vectors
        # that are time dependent
    
        act1 = np.ones((len(time), 1)) * 0.0
        act2 = np.ones((len(time), 1)) * 0.0
    
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
        res=sim.results()
        plt.plot(res[:, 1]*180/np.pi, res[:, 2])
    
        # In order to obtain internal paramters of the muscle
        # Check SystemSimulation.py results_muscles() method for more information
        #res_muscles = sim.results_muscles()

    # Plotting the results
    plt.xlabel('Position [deg]')
    plt.ylabel('Velocity [rad.s]')
    plt.legend(leg)
    plt.grid()
    
    
    #Asymmetric Attachment
    leg=[]
    
    plt.figure('3b_2')
    plt.title('Pendulum Phase')
    for i in range(0,len(a)):
        leg.append('Origins= 0.2m ; Attachment1= 0.2m ; Attachement2= {}m'.format(a[i]))
        
        # Define Muscle Attachment points
        m1_origin = np.array([-0.2, 0.0])  # Origin of Muscle 1
        m1_insertion = np.array([0.0, -0.2])  # Insertion of Muscle 1
    
        m2_origin = np.array([0.2, 0.0])  # Origin of Muscle 2
        m2_insertion = np.array([0.0, -a[i]])  # Insertion of Muscle 2
        
        # Attach the muscles
        muscles.attach(np.array([m1_origin, m1_insertion]),
                       np.array([m2_origin, m2_insertion]))
        
        # Muscle Model initial condition
        x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])

        x0 = np.concatenate((x0_P, x0_M))  # System initial conditions
    
        # Create a system with Pendulum and Muscles using the System Class
        # Check System.py for more details on System class
        sys = System()  # Instantiate a new system
        sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
        sys.add_muscle_system(muscles)  # Add the muscle model to the system
    
    
        ##### System Simulation #####
        # For more details on System Simulation check SystemSimulation.py
        # SystemSimulation is used to initialize the system and integrate
        # over time
    
        sim = SystemSimulation(sys)  # Instantiate Simulation object
    
        # Add muscle activations to the simulation
        # Here you can define your muscle activation vectors
        # that are time dependent
    
        act1 = np.ones((len(time), 1)) * 0.0
        act2 = np.ones((len(time), 1)) * 0.0
    
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
        res=sim.results()
        plt.plot(res[:, 1]*180/np.pi, res[:, 2])
    
        # In order to obtain internal paramters of the muscle
        # Check SystemSimulation.py results_muscles() method for more information
        #res_muscles = sim.results_muscles()

    # Plotting the results
    plt.xlabel('Position [deg]')
    plt.ylabel('Velocity [rad.s]')
    plt.legend(leg)
    plt.grid()
    
    """
    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        biolog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(res, pendulum, muscles)
    # To start the animation
    simulation.animate()
    """
    

if __name__ == '__main__':
    exercise3b()

