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
    return pendulum_system(state[0], state[1], *args, **kwargs)[:, 0]

def amplitude_experiments(parameters, x0_list, time):
    theta_amp = np.zeros(np.shape(x0_list)[0])    
    dtheta_amp = np.zeros(np.shape(x0_list)[0])
    for i,x0 in enumerate(x0_list):
        res = integrate(pendulum_integration, x0, time, args=(parameters,))
        thetas = np.array(res.state)[:,0]
        dthetas = np.array(res.state)[:,1]
        theta_amp[i] = np.max(thetas,axis=0)-np.min(thetas,axis=0)
        dtheta_amp[i] = np.max(dthetas,axis=0)-np.min(dthetas,axis=0)
    return [theta_amp/2,dtheta_amp/2]



def exercise1():
    """ Exercise 1  """
    biolog.info("Executing Lab 4 : Exercise 1")
    parameters = PendulumParameters()
    biolog.info(parameters.showParameters())

    # Simulation Parameters
    t_start = 0.0
    t_stop = 5.0
    dt = 0.001
    time = np.arange(t_start, t_stop, dt)
    
    parameters.b1 = 0.0
    parameters.b2 = 0.0
    
    #Question 1a
    temp = np.size(time)/3
    x0 = [0.0, 0.5]
    for i in range(3):
        res = integrate(pendulum_integration, x0, time[temp*i:temp*(i+1)], args=(parameters,))
        x0 = res.state[-1] + 0.05*np.random.random(2) #Introduce random perturbations by abruptly changing the state
        res.plot_state("State")
        res.plot_phase("Phase")
    
    #Question 1c
    x0_list = [[1.0, 0.0],[-1.0, 0.0],[1.0, 1.0],[1.0, -1.0],[1.5, 0.0]]
    parameters = PendulumParameters()
    parameters.b1 = 0.0
    parameters.b2 = 0.0
    K_range = np.arange(1,21,1)
    theta_amps = np.zeros((np.size(K_range),5))
    dtheta_amps = np.zeros((np.size(K_range),5))
    for i,k in enumerate(K_range):
        parameters.k1 = k
        theta_amps[i,:],dtheta_amps[i,:] = amplitude_experiments(parameters, x0_list, time)
    
    plt.figure()
    plt.plot(K_range,theta_amps*180/np.pi,linewidth=2)
    plt.grid()
    plt.legend([r'$x_0=[1.0, 0.0]$',r'$x_0=[-1.0, 0.0]$',r'$x_0=[1.0, 1.0]$',r'$x_0=[1.0, -1.0]$',r'$x_0=[1.5, 0.0]$'],
               bbox_to_anchor=(1.04,0.5), loc="center left")
#    plt.ylim(np.min(theta_amps.flatten()),np.max(theta_amps.flatten()))
    plt.title('Effect of changing one spring constant on the amplitude of oscillations',fontsize=12)
    plt.xlabel(r'$K_1$ [N/rad]')
    plt.ylabel('Amplitude of oscillation [degrees]')
    plt.show()
    
    plt.figure()
    plt.plot(K_range,dtheta_amps,linewidth=2)
    plt.grid()
    plt.legend([r'$x_0=[1.0, 0.0]$',r'$x_0=[-1.0, 0.0]$',r'$x_0=[1.0, 1.0]$',r'$x_0=[1.0, -1.0]$',r'$x_0=[1.5, 0.0]$'],
               bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.title('Effect of changing one spring constant on the amplitude of velocities',fontsize=12)
    plt.xlabel(r'$K_1$ [N/rad]')
    plt.ylabel('Amplitude of velocity [rad/s]')
    plt.show()
    
    
    parameters = PendulumParameters()
    parameters.b1 = 0.0
    parameters.b2 = 0.0
    theta_ref_range = np.linspace(0,np.pi/2,20)
    theta_amps = np.zeros((np.size(K_range),5))
    dtheta_amps = np.zeros((np.size(K_range),5))
    for i,theta_ref in enumerate(theta_ref_range):
        parameters.s_theta_ref1 = theta_ref
        theta_amps[i,:],dtheta_amps[i,:] = amplitude_experiments(parameters, x0_list, time)
    
    plt.figure()
    plt.plot(theta_ref_range*180/np.pi,theta_amps*180/np.pi,linewidth=2)
    plt.grid()
    plt.legend([r'$x_0=[1.0, 0.0]$',r'$x_0=[-1.0, 0.0]$',r'$x_0=[1.0, 1.0]$',r'$x_0=[1.0, -1.0]$',r'$x_0=[1.5, 0.0]$'],
               bbox_to_anchor=(1.04,0.5), loc="center left")
#    plt.ylim(np.min(theta_amps.flatten()),np.max(theta_amps.flatten()))
    plt.title('Effect of changing one reference angle on the amplitude of oscillations',fontsize=12)
    plt.xlabel(r'$\theta_{ref1}$ [degrees]')
    plt.ylabel('Amplitude of oscillation [degrees]')
    plt.show()
    
    plt.figure()
    plt.plot(theta_ref_range*180/np.pi,dtheta_amps,linewidth=2)
    plt.grid()
    plt.legend([r'$x_0=[1.0, 0.0]$',r'$x_0=[-1.0, 0.0]$',r'$x_0=[1.0, 1.0]$',r'$x_0=[1.0, -1.0]$',r'$x_0=[1.5, 0.0]$'],
               bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.title('Effect of changing one reference angle on the amplitude of velocities',fontsize=12)
    plt.xlabel(r'$\theta_{ref1}$ [degrees]')
    plt.ylabel('Amplitude of velocity [rad/s]')
    plt.show()
    
    x0 = [[1.5, 0.0]]
    
    range_k=np.arange(1,100,5)
    range_thetaref=np.arange(-np.pi/2,np.pi/2,0.1)
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

