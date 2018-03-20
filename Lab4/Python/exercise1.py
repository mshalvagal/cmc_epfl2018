""" Lab 4 """

import numpy as np
import matplotlib.pyplot as plt
from biopack import integrate, DEFAULT, parse_args
import biolog
from SystemParameters import PendulumParameters
from lab4_pendulum import pendulum_system

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]


def pendulum_integration(state, time, *args, **kwargs):
    """ Function for system integration """
    #biolog.warning("Pendulum equation with spring and damper must be implemented")  # l_S
    return pendulum_system(state[0], state[1], *args, **kwargs)[:, 0]


def exercise1():
    """ Exercise 1  """
    biolog.info("Executing Lab 4 : Exercise 1")
    parameters = PendulumParameters()
    biolog.info(
        "Find more information about Pendulum Parameters in SystemParameters.py")
    biolog.warning("Loading default pendulum parameters")
    biolog.info(parameters.showParameters())

    # Simulation Parameters
    t_start = 0.0
    t_stop = 5.0
    dt = 0.001
    #biolog.warning("Using large time step dt={}".format(dt))
    
    x0 = [-np.pi/16, 5.0]
    time = np.arange(t_start, t_stop, dt)
    #we add a perturbation 
    perturb = np.random.random(2)
    
    t_perturb = t_stop/2
    time_p1 = np.arange(t_start, t_perturb, dt)
    res_p1 = integrate(pendulum_integration, x0, time_p1, args=(parameters, ))
    res_p1.plot_state("State")
    res_p1.plot_phase("Phase")
    
    x0_p2 = res_p1.state[-1] + perturb
    time_p2 = np.arange(t_perturb, t_stop, dt)
    res_p2 = integrate(pendulum_integration, x0_p2, time_p2, args=(parameters, ))
    res_p2.plot_state("State")
    res_p2.plot_phase("Phase")
    
    range_k=np.arange(0,100,10)
    range_thetaref=np.arange(-np.pi/4,np.pi/4,0.1)
    amp_theta=np.zeros([len(range_k),len(range_thetaref)])
    max_speed =np.zeros([len(range_k),len(range_thetaref)])
    for i,k in enumerate(range_k):
        parameters.k1=k
        parameters.k2=k
        for j,theta_ref in enumerate(range_thetaref):
            parameters.s_theta_ref1=theta_ref
            parameters.s_theta_ref2=theta_ref
            res_temp = integrate(pendulum_integration, x0, time, args=(parameters, ))
            temp=np.array(res_temp.state)
            thetas=temp[:,0]
            dthetas=temp[:,1]
            amp_theta[i,j]=max(thetas)-min(thetas)
            max_speed[i,j]=max(dthetas)
    plt.figure()
    plt.contourf(range_thetaref,range_k,amp_theta)
    plt.xlabel("Theta ref")
    plt.ylabel("Spring constant K")
    plt.title("Amplitude")
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.contourf(range_thetaref,range_k,max_speed)
    plt.title("Max velocity")
    plt.xlabel("Theta ref")
    plt.ylabel("Spring constant K")
    plt.colorbar()
    plt.show()

    if DEFAULT["save_figures"] is False:
        plt.show()
    return


if __name__ == '__main__':
    exercise1()

