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


def exercise3a():
    """ Main function to run for Exercise 3.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    plt.close('all')
    #Question 3a
    m1_origin = np.array([-0.25, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.25])  # Insertion of Muscle 1

    m2_origin = np.array([0.25, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.25])  # Insertion of Muscle 2
    
    theta=np.linspace(-np.pi/2,np.pi/2)
    
    m1_length = np.sqrt(m1_origin[0]**2 + m1_insertion[1]**2 +
            2 * np.abs(m1_origin[0]) * np.abs(m1_insertion[1]) * np.sin(theta))
    m2_length = np.sqrt(m2_origin[0]**2 + m2_insertion[1]**2 +
            2 * np.abs(m2_origin[0]) * np.abs(m2_insertion[1]) * np.sin(-theta))
    
    plt.figure('Lengths')
    plt.title('Length of the muscle with respect to the position of the limb')
    plt.plot(theta*180/np.pi, m1_length)
    plt.plot(theta*180/np.pi, m2_length)
    plt.xlabel('Position [deg]')
    plt.ylabel('Muscle length [m]')
    plt.legend(['M1','M2'])
    plt.grid()
    
    m1_moment_arm= m1_origin[0] * m1_insertion[1] * np.cos(theta) / m1_length
    m2_moment_arm= m2_origin[0] * m2_insertion[1] * np.cos(-theta) / m2_length
    
    plt.figure('Moment Arms')
    plt.title('Moment arm over the muscle with respect to the position of the limb')
    plt.plot(theta*180/np.pi, m1_moment_arm)
    plt.plot(theta*180/np.pi, m2_moment_arm)
    plt.xlabel('Position [deg]')
    plt.ylabel('Moment arm [m]')
    plt.legend(['M1','M2'])
    plt.grid()
    
    #Varying the attachement points

    #a1s=np.linspace(-0.5,-0.1,5)
    a2s=np.linspace(-0.5,-0.1,5)
    a1=-0.2 #completely symmetrical behavior
    
    m_lengths=np.zeros((len(a2s),len(theta)))
    m_moment_arms=np.zeros((len(a2s),len(theta)))
    leg=[]
    for i in range(0,len(a2s)):
        m_lengths[i,:]=np.sqrt(a1**2 + a2s[i]**2 +
                         2 * np.abs(a1) * np.abs(a2s[i]) * np.sin(theta))
        m_moment_arms[i,:]= a1 * a2s[i] * np.cos(theta) / m_lengths[i,:]
        leg.append('Origin: {}m, Insertion: {}m'.format(a1,a2s[i]))
        
#    for i in range(0,len(a2s)):
#        m_lengths[i+len(a1s),:]=np.sqrt(a1s[1]**2 + a2s[i]**2 +
#                         2 * np.abs(a1s[1]) * np.abs(a2s[i]) * np.sin(theta))
#        leg.append('Origin: {}m, Insertion: {}m'.format(a1s[1],a2s[i]))
    plt.figure('3a_L')
    plt.title('Length of M1 with respect to the position of the limb') 
    for i in range(0,len(m_lengths)):
        plt.plot(theta*180/np.pi, m_lengths[i,:])
    plt.plot((theta[0]*180/np.pi,theta[len(theta)-1]*180/np.pi),(0.11,0.11), ls='dashed')
    leg.append('l_opt')
    plt.plot((theta[0]*180/np.pi,theta[len(theta)-1]*180/np.pi),(0.13,0.13), ls='dashed')
    leg.append('l_slack') 
    plt.xlabel('Position [deg]')
    plt.ylabel('Muscle length [m]')
    plt.legend(leg)
    plt.grid()
    
    plt.figure('3a_MA')
    plt.title('Moment arm over M1 with respect to the position of the limb')
    for i in range(0,len(m_moment_arms)):
        plt.plot(theta*180/np.pi, m_moment_arms[i,:])
    plt.xlabel('Position [deg]')
    plt.ylabel('Moment arm [m]')
    plt.legend(leg)
    plt.grid()

if __name__ == '__main__':
    exercise3a()

