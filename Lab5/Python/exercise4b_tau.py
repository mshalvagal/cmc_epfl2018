""" Lab 5 Exercise 1

This file implements the pendulum system with two muscles attached

"""

from SystemParameters import (
    PendulumParameters,
    MuscleParameters,
    NetworkParameters
)
from Muscle import Muscle
import numpy as np
import biolog
from matplotlib import pyplot as plt
from biopack import DEFAULT
from biopack.plot import save_figure
from PendulumSystem import Pendulum
from MuscleSystem import MuscleSytem
from NeuralSystem import NeuralSystem
from SystemSimulation import SystemSimulation
from SystemAnimation import SystemAnimation
from System import System

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def exercise4b_tau():
    """ Main function to run for Exercise 4.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    plt.close('all')
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
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))
    
    ##### Time #####
    t_max = 5.  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([np.pi / 4., 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.l_CE, 0., M2.l_CE])

    x0_N = np.array([-0.5, 1, 0.5, 1])  # Neural Network Initial Conditions
    x0_N = np.array([-2, 2, 4, -4])  # Neural Network Initial Conditions

    x0 = np.concatenate((x0_P, x0_M, x0_N))  # System initial conditions
    
    ##### Neural Network #####
    # The network consists of four neurons
    N_params = NetworkParameters()  # Instantiate default network parameters
    w_0=np.transpose([[0., -10., 10., -10.],
                      [-10., 0., -10., 10.],
                      [-10., 0., 0., 0.],
                      [0., -10., 0., 0.]] )
    N_params.w=w_0
    tau_0=N_params.tau    
    
    if False:
        #3.1 Play with scaling the tau
        scale=np.array([0.2,1,5])

        #Figures
        fig1=plt.figure('Phase')
        ax1=fig1.add_subplot(111)
        fig2=plt.figure('Neurons_MP')
        
        leg1=[]
        for i in range(0,len(scale)):
            s=scale[i]
            N_params.tau = tau_0*s
            
            # Create a new neural network with above parameters
            neural_network = NeuralSystem(N_params)
    
            # Create system of Pendulum, Muscles and neural network using SystemClass
            # Check System.py for more details on System class
            sys = System()  # Instantiate a new system
            sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
            sys.add_muscle_system(muscles)  # Add the muscle model to the system
            # Add the neural network to the system
            sys.add_neural_system(neural_network)
           
            ##### System Simulation #####
            # For more details on System Simulation check SystemSimulation.py
            # SystemSimulation is used to initialize the system and integrate
            # over time
    
            sim = SystemSimulation(sys)  # Instantiate Simulation object
    
            sim.initalize_system(x0, time)  # Initialize the system state
    
            # Integrate the system for the above initialized state and time
            sim.simulate()
    
            # Obtain the states of the system after integration
            # res is np.array [time, states]
            # states vector is in the same order as x0
            res = sim.results()
    
            #   In order to obtain internal paramters of the muscle
            # Check SystemSimulation.py results_muscles() method for more information
            res_muscles = sim.results_muscles()
    
            # Plotting the results
            ax1.plot(res[:, 1], res[:, 2])
            leg1.append('Tau scaled with {}'.format(scale[i]))
            ax2=fig2.add_subplot(len(scale),1,i+1)
            leg2=[]
            for j in range(1,5):
                ax2.plot(res[:, 0], res[:, j+6])
                leg2.append('Neuron{}'.format(j))
            
            ax2.set_title('Neurons Membrane Potential ; tau scaled with {}'.format(scale[i]))    
            ax2.set_ylabel('Membrane Potential [mV]')
            ax2.set_xlabel('Time [s]')
            fig2.legend(leg2)

            
        ax1.set_title('Pendulum Phase')
        ax1.set_xlabel('Position [rad]')
        ax1.set_ylabel('Velocity [rad.s]')
        fig1.legend(leg1)
        
        fig1.show()
    
        fig2.show()
        
    if False:
        #3.2 Play with scaling the tau of neuron 1-2 VS 3-4
        scale=np.array([0.2,1,5])

        #Figures
        fig1=plt.figure('Phase')
        ax1=fig1.add_subplot(111)
        fig2=plt.figure('Neurons_MP')
        
        leg1=[]
        for i in range(0,len(scale)):
            s=[1,1,scale[i],scale[i]]
            N_params.tau = tau_0*s
            
            # Create a new neural network with above parameters
            neural_network = NeuralSystem(N_params)
    
            # Create system of Pendulum, Muscles and neural network using SystemClass
            # Check System.py for more details on System class
            sys = System()  # Instantiate a new system
            sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
            sys.add_muscle_system(muscles)  # Add the muscle model to the system
            # Add the neural network to the system
            sys.add_neural_system(neural_network)
           
            ##### System Simulation #####
            # For more details on System Simulation check SystemSimulation.py
            # SystemSimulation is used to initialize the system and integrate
            # over time
    
            sim = SystemSimulation(sys)  # Instantiate Simulation object
    
            sim.initalize_system(x0, time)  # Initialize the system state
    
            # Integrate the system for the above initialized state and time
            sim.simulate()
    
            # Obtain the states of the system after integration
            # res is np.array [time, states]
            # states vector is in the same order as x0
            res = sim.results()
    
            #   In order to obtain internal paramters of the muscle
            # Check SystemSimulation.py results_muscles() method for more information
            res_muscles = sim.results_muscles()
    
            # Plotting the results
            ax1.plot(res[:, 1], res[:, 2])
            leg1.append('Tau scaled with {}'.format(scale[i]))
            ax2=fig2.add_subplot(len(scale),1,i+1)
            leg2=[]
            for j in range(1,5):
                ax2.plot(res[:, 0], res[:, j+6])
                leg2.append('Neuron{}'.format(j))
            
            ax2.set_title('Neurons Membrane Potential ; tau of neuron 3 and 4 scaled with {}'.format(scale[i]))    
            ax2.set_ylabel('Membrane Potential [mV]')
            ax2.set_xlabel('Time [s]')
            fig2.legend(leg2)

            
        ax1.set_title('Pendulum Phase')
        ax1.set_xlabel('Position [rad]')
        ax1.set_ylabel('Velocity [rad.s]')
        fig1.legend(leg1)
        
        fig1.show()
    
        fig2.show()
        
    if True:
        #3.3 Play with scaling the tau of neuron 1-3 VS 2-4
        scale=np.array([0.2,1,5])

        #Figures
        fig1=plt.figure('Phase')
        ax1=fig1.add_subplot(111)
        fig2=plt.figure('Neurons_MP')
        
        leg1=[]
        for i in range(0,len(scale)):
            s=[1,scale[i],1,scale[i]]
            N_params.tau = tau_0*s
            
            # Create a new neural network with above parameters
            neural_network = NeuralSystem(N_params)
    
            # Create system of Pendulum, Muscles and neural network using SystemClass
            # Check System.py for more details on System class
            sys = System()  # Instantiate a new system
            sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
            sys.add_muscle_system(muscles)  # Add the muscle model to the system
            # Add the neural network to the system
            sys.add_neural_system(neural_network)
           
            ##### System Simulation #####
            # For more details on System Simulation check SystemSimulation.py
            # SystemSimulation is used to initialize the system and integrate
            # over time
    
            sim = SystemSimulation(sys)  # Instantiate Simulation object
    
            sim.initalize_system(x0, time)  # Initialize the system state
    
            # Integrate the system for the above initialized state and time
            sim.simulate()
    
            # Obtain the states of the system after integration
            # res is np.array [time, states]
            # states vector is in the same order as x0
            res = sim.results()
    
            #   In order to obtain internal paramters of the muscle
            # Check SystemSimulation.py results_muscles() method for more information
            res_muscles = sim.results_muscles()
    
            # Plotting the results
            ax1.plot(res[:, 1], res[:, 2])
            leg1.append('Tau scaled with {}'.format(scale[i]))
            ax2=fig2.add_subplot(len(scale),1,i+1)
            leg2=[]
            for j in range(1,5):
                ax2.plot(res[:, 0], res[:, j+6])
                leg2.append('Neuron{}'.format(j))
            
            ax2.set_title('Neurons Membrane Potential ; tau of neuron 2 and 4 scaled with {}'.format(scale[i]))    
            ax2.set_ylabel('Membrane Potential [mV]')
            ax2.set_xlabel('Time [s]')
            fig2.legend(leg2)

            
        ax1.set_title('Pendulum Phase')
        ax1.set_xlabel('Position [rad]')
        ax1.set_ylabel('Velocity [rad.s]')
        fig1.legend(leg1)
        
        fig1.show()
    
        fig2.show()
        
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
    simulation = SystemAnimation(res, pendulum, muscles, neural_network)
    # To start the animation
    simulation.animate()
"""

if __name__ == '__main__':
    exercise4b_tau()

