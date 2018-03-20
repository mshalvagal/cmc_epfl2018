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

def amplitude_experiments(parameters, x0, time):
    res = integrate(pendulum_integration, x0, time, args=(parameters,))
    thetas = np.array(res.state)[:,0]
    dthetas = np.array(res.state)[:,1]
    theta_amp = np.max(thetas,axis=0)-np.min(thetas,axis=0)
    dtheta_amp = np.max(dthetas,axis=0)-np.min(dthetas,axis=0)
    return [theta_amp,dtheta_amp]



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
    time = np.arange(t_start, t_stop, dt)
    
    parameters.b1=2
    parameters.b2=2
    
    #Question 1a
    temp = np.size(time)/3
    x0 = [0.0, 0.5]
    for i in range(3):
        res = integrate(pendulum_integration, x0, time[temp*i:temp*(i+1)], args=(parameters,))
        x0 = res.state[-1] + 0.05*np.random.random(2)
        res.plot_state("State")
        res.plot_phase("Phase")
    
    #Question 1b
    K_range = np.arange(1,21,1)
    theta_amps = np.zeros((np.size(K_range),4))
    dtheta_amps = np.zeros((np.size(K_range),4))
    for i,k in enumerate(K_range):
        parameters.k1 = k
        
        x0 = [1.0, 0.0]
        theta_amps[i,0],dtheta_amps[i,0] = amplitude_experiments(parameters, x0, time)
        
        x0 = [-1.0, 0.0]
        theta_amps[i,1],dtheta_amps[i,1] = amplitude_experiments(parameters, x0, time)
        
        x0 = [1.0, 1.0]
        theta_amps[i,2],dtheta_amps[i,2] = amplitude_experiments(parameters, x0, time)
        
        x0 = [1.0, -1.0]
        theta_amps[i,3],dtheta_amps[i,3] = amplitude_experiments(parameters, x0, time)
                
        x0 = [2.0, 0.0]
        theta_amps[i,3],dtheta_amps[i,3] = amplitude_experiments(parameters, x0, time)
    
    plt.figure(3)
    plt.plot(K_range,theta_amps)
    plt.legend([r'+$\theta_1$',r'-$\theta$',r'+$d\theta$',r'-$d\theta$',r'++$d\theta$'])
    
    plt.figure(4)
    plt.plot(K_range,dtheta_amps)
    plt.legend([r'+$\theta$',r'-$\theta$',r'+$d\theta$',r'-$d\theta$',r'++$d\theta$'])
    
    #biolog.warning("Using large time step dt={}".format(dt))
    
    x0 = [-np.pi/16, 5.0]
    
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
            [amp_theta[i,j],max_speed[i,j]]=amplitude_experiments(parameters,x0,time)
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

