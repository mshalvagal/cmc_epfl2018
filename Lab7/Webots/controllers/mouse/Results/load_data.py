""" Load results data """

import numpy as np
import matplotlib.pyplot as plt

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=15.0)
plt.rc('axes', titlesize=20.0)     # fontsize of the axes title
plt.rc('axes', labelsize=15.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15.0)    # fontsize of the tick labels


def load_data():
    """ Load results data """
    ankle_l_trajectory = np.load("ankle_l_trajectory.npy")
    ankle_r_trajectory = np.load("ankle_r_trajectory.npy")
    return [ankle_l_trajectory, ankle_r_trajectory]

def plot_trajectories():
    """ Plot the trajectories of the hind feet"""
    
    ankle_l_trajectory, ankle_r_trajectory = load_data()
    
    time=np.linspace(1,len(ankle_l_trajectory[:,0]),len(ankle_l_trajectory[:,0]));
    
    plt.figure('Trajectories')
    plt.subplot(311)
    plt.plot(time,ankle_l_trajectory[:,0])
    plt.plot(time,ankle_r_trajectory[:,0])
    plt.title('Trajectory of the X component')
    #plt.xlabel('Time [ms]')
    plt.ylabel('Position [cm]')
    plt.legend(['Left ankle','Right ankle'],loc='upper right')
    
    plt.subplot(312)
    plt.plot(time,ankle_l_trajectory[:,1])
    plt.plot(time,ankle_r_trajectory[:,1])
    plt.title('Trajectory of the Y component')
    #plt.xlabel('Time [ms]')
    plt.ylabel('Position [cm]')
    plt.legend(['Left ankle','Right ankle'],loc='upper right')
    
    plt.subplot(313)
    plt.plot(time,ankle_l_trajectory[:,2])
    plt.plot(time,ankle_r_trajectory[:,2])
    plt.title('Trajectory of the Z component')
    plt.xlabel('Time [ms]')
    plt.ylabel('Position [cm]')
    plt.legend(['Left ankle','Right ankle'],loc='upper right')
    
    plt.tight_layout()
    
#    plt.suptitle('Decomposition of the trajectories of the hind feet')
    return


def main():
    """ Main """
    
    #ankle_l_trajectory, ankle_r_trajectory = load_data()
    #print("ankle_l_trajectory:\n{}\nankle_r_trajectory:\n{}".format(
    #    ankle_l_trajectory,
    #    ankle_r_trajectory
    #))
    
    plot_trajectories()
    
    return


if __name__ == '__main__':
    main()
