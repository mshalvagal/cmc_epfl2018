""" Lab 4 - Exercise 2 """

import numpy as np
import matplotlib.pyplot as plt
from biopack import integrate, DEFAULT, parse_args
from biopack.plot import save_figure
from SystemParameters import MuscleParameters, MassParameters
from lab4_mass import mass_system
import biolog
from scipy.integrate import odeint

from operator import add
# Import muscule model
import Muscle

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

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


def mass_integration(state, time, *args):
    """ Function to integrate muscle-mass system """
    force = args[0]
    mass_parameters = args[1]
    return mass_system(state[0], state[1], force, mass_parameters)


def muscle_integrate(muscle, deltaLength, activation, dt):
    """ Function steps or integrates the muscle model by the specified time_step
    dt.

    Parameters:
    -----
        muscle : <Muscle>
            Instance of Muscle class
        deltaLength : float
            Change in Muscle Tendon Length
        activation : float
            Activation of the muscle
        dt : float
            Time step to integrate (Good value is 0.001)

    Returns:
    --------
        res : dict

        res['l_CE'] :
            Contracticle element length
        res['v_CE'] :
            Contracticle element velocity
        res['l_MTC'] :
            Length of muscle tendon unit
        res['activeForce'] :
            Contracticle element force
        res['passiveForce'] :
            Passive element force
        res['force'] :
            activeForce + passiveForce
        res['tendonForce'] :
            Muscle tendon Force

    Example:
    ========
         >>> res = muscle_integrate(muscle, deltaLength=0.0, activation=0.05,
                                    dt=0.01)
    """
    muscle.stim = activation
    muscle.deltaLength = deltaLength
    muscle.step(dt)
    res = {}
    res['l_CE'] = muscle.l_CE
    res['v_CE'] = muscle.v_CE
    res['l_MTC'] = muscle.l_MTC
    res['activeForce'] = muscle.activeForce
    res['passiveForce'] = muscle.passiveForce
    res['force'] = muscle.force
    res['tendonForce'] = muscle.tendonForce
    return res


def isometric_contraction(muscle, stretch=np.arange(0.0, 0.05, 0.01),
                          activation=0.05):
    """ This function implements the isometric contraction
    of the muscle.

    Parameters:
    -----------
        muscle : <Muscle>
            Instance of Muscle class
        stretch : list/array
            A list/array of muscle stretches to be evaluated
        activation : float
            Muscle activation

    Returns:
    -------
    """
    stretch = np.array(stretch)

    # Time settings
    t_start = 0.0  # Start time
    t_stop = 0.2  # Stop time
    dt = 0.001  # Time step
    time = np.arange(t_start,t_stop,dt)
    active_force = []
    passive_force = []
    CE_length = []
    for i,stretch_value in enumerate(stretch):
        for  j in enumerate(time):
            res = muscle_integrate(muscle, stretch_value, activation,dt)
        active_force.append(res['activeForce'])
        passive_force.append(res['passiveForce'])
        CE_length.append(res['l_CE'])
        
    return active_force, passive_force, CE_length


def isotonic_contraction(muscle, activation, load=np.arange(1., 100, 10),
                         muscle_parameters=MuscleParameters(),
                         mass_parameters=MassParameters()):
    """ This function implements the isotonic contraction
    of the muscle.

    Parameters:
    -----------
        muscle : <Muscle>
            Instance of Muscle class
        load : list/array
            External load to be applied on the muscle.
            It is the mass suspended by the muscle
        muscle_parameters : MuscleParameters
            Muscle paramters instance
        mass_paramters : MassParameters
            Mass parameters instance


    Since the muscle model is complex and sensitive to integration,
    use the following example as a hint to complete your implementation.

    Example:
    --------

    >>> for load_ in load:
    >>>    # Iterate over the different muscle stretch values
    >>>    mass_parameters.mass = load_ # Set the mass applied on the muscle
    >>>    state = np.copy(x0) # Reset the state for next iteration
    >>>    for time_ in time:
    >>>         # Integrate for 0.2 seconds
    >>>        # Integration before the quick release
    >>>        res = muscle_integrate(muscle, state[0], activation=1.0, dt)
    >>>    for time_ in time:
    >>>        # Quick Release experiment
    >>>        # Integrate the mass system by applying the muscle force on to it
    >>>        for a time step dt
    >>>             mass_res = odeint(mass_integration, state,  [
    >>>             time_, time_ + dt], args=(muscle.force, load_, mass_parameters))
    >>>             state[0] = mass_res[-1, 0] # Update state with final postion of
    >>>             mass
    >>>             state[1] = mass_res[-1, 1] # Update state with final position of
    >>>             velocity
    >>>        # Now update the muscle model with new position of mass
    >>>        res = muscle_integrate(muscle, state[0], 1.0, dt)
    >>>        # Save the relevant data
    >>>        res_[id] = res['v_CE']
    >>>        if(res['l_MTC'] > muscle_parameters.l_opt + muscle_parameters.l_slack):
    >>>             velocity_ce[idx] = min(res_[:])
    >>>         else:
    >>>             velocity_ce[idx] = max(res_[:])

    """
    load = np.array(load)
    velocity_ce=np.zeros(len(load))

    #biolog.warning('Exercise 2b isotonic contraction to be implemented')

    # Time settings
    t_start = 0.0  # Start time
    t_stop = 0.2  # Stop time
    my_dt = 0.0005  # Time step
    time = np.arange(t_start,t_stop,my_dt)

    x0 = np.array([0.0, 0.0])  # Initial state of the mass (position, velocity)
    for idx,load_ in enumerate(load):
       # Iterate over the different muscle stretch values
        mass_parameters.mass = load_ # Set the mass applied on the muscle
        state = np.copy(x0) # Reset the state for next iteration
        
        for time_ in time:
             # Integrate for 0.2 seconds
            # Integration before the quick release
            res = muscle_integrate (muscle, state[0], activation,  my_dt)
        
        res_vCE=np.zeros(len(time))    
        for index,time_ in enumerate(time):
            # Quick Release experiment
            # Integrate the mass system by applying the muscle force on to it for a time step dt
            mass_res = odeint(mass_integration, state,  [time_, time_ + my_dt], 
                                   args=(muscle.force, mass_parameters))
            state[0] = mass_res[-1, 0] # Update state with final postion of mass
            state[1] = mass_res[-1, 1] # Update state with final position of velocity
            # Now update the muscle model with new position of mass
            res = muscle_integrate(muscle, state[0], activation, my_dt)
            # Save the relevant data
            res_vCE[index] = res['v_CE']
            
        if(res['l_MTC'] > muscle_parameters.l_opt + muscle_parameters.l_slack):
            #the muscle length at the end is superior to its"initial length", it is stretched
            #and has not been able to pull the mass up. It has been lengthening and the max
            #speed is negative            
            velocity_ce[idx] = min(res_vCE[:])
        else:
            #the muscle has enough force to hold the mass, its has pulled up the mass
            # and the max velocity is positive.
            velocity_ce[idx] = max(res_vCE[:])
    return velocity_ce


def exercise2a():
    """ Exercise 2a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    biolog.warning("Loading default muscle parameters")
    biolog.info(parameters.showParameters())

    # Create muscle object
    muscle = Muscle.Muscle(parameters)
    #biolog.warning("Isometric muscle contraction to be implemented")
        
    #Question 2a
    activation = 0.5
    stretch=np.arange(0.0, 0.05, 0.001)      
    active_force, passive_force, CE_length = isometric_contraction(muscle,stretch,activation)
    
    #plt.figure()
    #plt.plot(stretch,active_force,stretch,passive_force)
    #plt.legend(['Active Force, activation = {}'.format(activation),'Passive Force'])
    #plt.xlabel('stretch')
    #plt.ylabel('Forces in N')
    #plt.show()
    
    #plt.figure()
    #plt.plot(stretch,CE_length)
    #plt.xlabel('Stretch')
    #plt.ylabel('Length of contractile element')
    #plt.show()
    
    plt.figure()
    plt.plot(stretch,active_force,stretch,passive_force)
    plt.plot(stretch,map(add,active_force,passive_force))
    plt.legend(['Active Force, activation = {}'.format(activation),'Passive Force','Total Force'],
                bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.xlabel('Absolute Stretch in meters')
    plt.ylabel('Forces in N')
    plt.savefig('2_a.png')
    plt.show()
 
    #Question 2b
    plt.figure()
    activation_range = np.arange(1,-0.1,-0.25)
    legend_list =[]
    for i, activation in enumerate(activation_range):
        if (activation<0.01):
            activation=0
        active_force,passive_force,CE_length = isometric_contraction(muscle,stretch,activation)
        plt.plot(stretch,active_force)
        legend_list.append('Active Force, activation = {0:.1g}'.format(activation))
    plt.plot(stretch,passive_force)
    legend_list.append('Passive Force')
    plt.legend(legend_list,bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.xlabel('Stretch')
    plt.ylabel('Forces in N')
    plt.savefig('2_b.png')
    plt.show()

    #Question 2c
    #we plot it with relative stretch and not absolute stretch.
    l_opt_range=np.array([0.01,0.1,0.5]) #0.11 is the default fiber length
    relative_stretch=np.arange(0,1,0.02)
    #relative_stretch relatively to l_opt
    legend_list_2c =[]
    plt.figure()
    for i, l_opt in enumerate(l_opt_range):
        muscle.l_opt = l_opt
        stretch=l_opt*relative_stretch
        active_force,passive_force,CE_length = isometric_contraction(muscle,stretch,0.5)
        plt.plot(relative_stretch*100,active_force,relative_stretch*100,passive_force)
        legend_list_2c.append(r'Active Force, l_opt = {} cm'.format(l_opt*100))
        legend_list_2c.append(r'Passive Force, l_opt = {} cm'.format(l_opt*100))
    plt.legend(legend_list_2c,bbox_to_anchor=(1.04,0.5),loc="center left")
    plt.xlabel('Relative Stretch in %')
    plt.ylabel('Forces in N')
    plt.ylim([0,1250])
    plt.savefig('2_c.png')
    plt.show()  

def exercise2b():
    """ Exercise 2b
    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest"""

    # Defination of muscles
    my_muscle_parameters = MuscleParameters()
    print(my_muscle_parameters.showParameters())

    my_mass_parameters = MassParameters()
    print(my_mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle.Muscle(my_muscle_parameters)
    
    #Question 2d
    my_loads = np.arange(1., 270, 1)
    activation = 1.0
    CE_velocity = isotonic_contraction(muscle, activation, load=my_loads, 
                                       muscle_parameters=my_muscle_parameters, 
                                       mass_parameters=my_mass_parameters)

    plt.figure()
    plt.plot(CE_velocity,my_loads)
    plt.xlabel('Contractile Element max velocity in m/s')
    plt.axhline(y=my_muscle_parameters.f_max/my_mass_parameters.g, 
                linewidth=1,color='k', linestyle='--')
    plt.axvline(x=0,linewidth=1,color='k',linestyle='--')
    plt.ylabel('Load applied in kg')
    plt.savefig('2_d.png')
    plt.show()      
    
    #Question 2f
    activation_2 = 0.5
    my_loads_2 = np.arange(1., 165, 1)
    CE_velocity_2 = isotonic_contraction(muscle, activation_2, load=my_loads_2, 
                                       muscle_parameters=my_muscle_parameters, 
                                       mass_parameters=my_mass_parameters)
    
    activation_3 = 0.1
    my_loads_3 = np.arange(1., 70, 1)
    CE_velocity_3 = isotonic_contraction(muscle, activation_3, load=my_loads_3, 
                                       muscle_parameters=my_muscle_parameters, 
                                       mass_parameters=my_mass_parameters)

    plt.figure()
    plt.plot(CE_velocity,my_loads)
    plt.plot(CE_velocity_2,my_loads_2)
    plt.plot(CE_velocity_3,my_loads_3)
    plt.legend(['For activation = {}'.format(activation),
                'For activation = {}'.format(activation_2),
                'For activation = {}'.format(activation_3)])
    plt.xlabel('Contractile Element max velocity in m/s')
    plt.ylabel('Load applied in kg')
    plt.savefig('2_f.png')
    plt.show()  


def exercise2():
    """ Exercise 2 """
    exercise2a()
    exercise2b()


if __name__ == '__main__':
    exercise2()

