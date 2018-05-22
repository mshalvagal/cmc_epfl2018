""" Load results data """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_data():
    """ Load results data """
    time = np.load("time.npy")
    ankle_l_trajectory = np.load("ankle_l_trajectory.npy")
    ankle_r_trajectory = np.load("ankle_r_trajectory.npy")
    foot_l_contact = np.load("foot_l_contact.npy")
    foot_r_contact = np.load("foot_r_contact.npy")
    muscle_lh_activations = np.load("muscle_lh_activations.npy")
    muscle_rh_activations = np.load("muscle_rh_activations.npy")
    muscle_lh_forces = np.load("muscle_lh_forces.npy")
    muscle_rh_forces = np.load("muscle_rh_forces.npy")
    joint_lh_positions = np.load("joint_lh_positions.npy")
    joint_rh_positions = np.load("joint_rh_positions.npy")
    return [time,
            ankle_l_trajectory,
            ankle_r_trajectory,
            foot_l_contact,
            foot_r_contact,
            muscle_lh_activations,
            muscle_rh_activations,
            muscle_lh_forces,
            muscle_rh_forces,
            joint_lh_positions,
            joint_rh_positions]

# Plotting ground contact gait


def y_options(**kwargs):
    """ Return y options """
    y_size = kwargs.pop("y_size", 4)
    y_sep = 1.0 / (y_size + 1)

    def y_pos(y): return (y + (y + 1) * y_sep) / (y_size + 1)

    return y_size, y_sep, y_pos


def add_patch(ax, x, y, **kwargs):
    """ Add patch """
    y_size, y_sep, y_pos = y_options(**kwargs)
    width = kwargs.pop("width", 1)
    height = kwargs.pop("height", y_sep)
    ax.add_patch(
        patches.Rectangle(
            (x, y_pos(y)),
            width,
            height,
            hatch='\\' if (y % 2) else '/'
        )
    )
    return


def plot_gait(time, gait, dt, **kwargs):
    """ Plot gait """
    figurename = kwargs.pop("figurename", "gait")
    fig1 = plt.figure(figurename)
    ax1 = fig1.add_subplot("111", aspect='equal')
    for t, g in enumerate(gait):
        for l, gait_l in enumerate(g):
            if gait_l:
                add_patch(ax1, time[t], l, width=dt, y_size=len(gait[0, :]))
    y_values = kwargs.pop(
        "y_values",
        [
            "Left\nFoot",
            "Right\nFoot",
            "Left\nHand",
            "Right\nHand"
        ][:len(gait[0, :])]
    )
    _, y_sep, y_pos = y_options(y_size=len(gait[0, :]))
    y_axis = [y_pos(y) + 0.5 * y_sep for y in range(4)]
    plt.yticks(y_axis, y_values)
    plt.xlabel("Time [s]")
    plt.ylabel("Gait")
    plt.axis('auto')
    plt.grid(True)
    return

def plot_trajectories_XYZ(t_start,t_stop):
    """ Plot the trajectories of the hind feet"""
    
    time, ankle_l_trajectory, ankle_r_trajectory,foot_l_contact,foot_r_contact,muscle_lh_activations, muscle_rh_activations,muscle_lh_forces,muscle_rh_forces,joint_lh_positions,joint_rh_positions = load_data()
   
    index_start = np.where(time == t_start)[0][0]
    index_end = np.where(time == t_stop)[0][0]
    
    time = time[index_start:index_end+1]
    ankle_l_trajectory = ankle_l_trajectory[index_start:index_end+1,:]
    ankle_r_trajectory = ankle_r_trajectory[index_start:index_end+1,:]
 
    #time=np.linspace(1,len(ankle_l_trajectory[:,0]),len(ankle_l_trajectory[:,0]));
    
    plt.figure('Trajectories')
    plt.subplot(311)
    plt.plot(time,ankle_l_trajectory[:,0])
    plt.plot(time,ankle_r_trajectory[:,0])
    #plt.title('Trajectory of the X component')
    plt.xlabel('Time [s]')
    plt.ylabel('X Position [cm]')
    plt.legend(['Left ankle','Right ankle'],loc='upper right')
    
    plt.subplot(312)
    plt.plot(time,ankle_l_trajectory[:,1])
    plt.plot(time,ankle_r_trajectory[:,1])
    #plt.title('Trajectory of the Y component')
    plt.xlabel('Time [s]')
    plt.ylabel('Y Position [cm]')
    plt.legend(['Left ankle','Right ankle'],loc='upper right')
    
    plt.subplot(313)
    plt.plot(time,ankle_l_trajectory[:,2])
    plt.plot(time,ankle_r_trajectory[:,2])
    #plt.title('Trajectory of the Z component')
    plt.xlabel('Time [s]')
    plt.ylabel('Z Position [cm]')
    plt.legend(['Left ankle','Right ankle'],loc='upper right')
        
#    plt.suptitle('Decomposition of the trajectories of the hind feet')
    return

def plot_muscle_activations(side,t_start,t_stop):
    """ Plot the trajectories of the hind feet"""
    
    time, ankle_l_trajectory, ankle_r_trajectory,foot_l_contact,foot_r_contact,muscle_lh_activations, muscle_rh_activations,muscle_lh_forces,muscle_rh_forces,joint_lh_positions,joint_rh_positions = load_data()

    index_start = np.where(time == t_start)[0][0]
    index_end = np.where(time == t_stop)[0][0]
    
    time = time[index_start:index_end+1]
    muscle_rh_activations = muscle_rh_activations[index_start:index_end+1,:]
    muscle_lh_activations = muscle_lh_activations[index_start:index_end+1,:]
    
    #time=np.linspace(1,len(ankle_l_trajectory[:,0]),len(ankle_l_trajectory[:,0]));
    if side =='right':
        muscle_activations = muscle_rh_activations
    elif side == 'left':
        muscle_activations = muscle_lh_activations       
    else:
        return
  
    plt.figure('Muscle activations')
    plt.subplot(241)
    plt.plot(time,muscle_activations[:,0])
    plt.title('Muscle PMA')
    #plt.xlabel('Time [s]')
    plt.ylabel('Muscle activation')

    plt.subplot(242)
    plt.plot(time,muscle_activations[:,1])
    plt.title('Muscle CF')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Muscle activation') 

    plt.subplot(243)    
    plt.plot(time,muscle_activations[:,2])
    plt.title('Muscle SM')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Muscle activation')
    
    plt.subplot(244)    
    plt.plot(time,muscle_activations[:,3])
    plt.title('Muscle POP')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Muscle activation')    
    
    plt.subplot(245)    
    plt.plot(time,muscle_activations[:,4])
    plt.title('Muscle RF')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle activation')  

    plt.subplot(246)    
    plt.plot(time,muscle_activations[:,5])
    plt.title('Muscle TA')
    plt.xlabel('Time [s]')
    #plt.ylabel('Muscle activation')    
    
    plt.subplot(247)    
    plt.plot(time,muscle_activations[:,6])
    plt.title('Muscle SOL')
    plt.xlabel('Time [s]')
    #plt.ylabel('Muscle activation')  
    
    plt.subplot(248)    
    plt.plot(time,muscle_activations[:,7])
    plt.title('Muscle LG')
    plt.xlabel('Time [s]')
    #plt.ylabel('Muscle activation')  
#    plt.suptitle('Decomposition of the trajectories of the hind feet')
    
    plt.suptitle('Muscle activations of the '+ side + ' limb')
    plt.show()
    return


def plot_joint_angles(t_start,t_stop):
    """ to plot the joint angles"""
    [time,
     ankle_l_trajectory,
     ankle_r_trajectory,
     foot_l_contact,
     foot_r_contact,
     muscle_lh_activations,
     muscle_rh_activations,
     muscle_lh_forces,
     muscle_rh_forces,
     joint_lh_positions,
     joint_rh_positions] = load_data()
    
    index_start = np.where(time == t_start)[0][0]
    index_end = np.where(time == t_stop)[0][0]
    
    time_plot = time[index_start:index_end+1]
    joint_lh_positions = joint_lh_positions[index_start:index_end+1,:]
    joint_rh_positions = joint_rh_positions[index_start:index_end+1,:]

    # Example to plot joint trajectories.
    # Feel free to change or use your own plot tools
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time_plot, np.rad2deg(joint_lh_positions[:, 0]))
    plt.plot(time_plot, np.rad2deg(joint_rh_positions[:, 0]))
    plt.ylabel('Hip Angle [deg]')
    plt.legend(['Left','Right'],loc='upper right')
    plt.grid('on')
    plt.subplot(3,1,2)
    plt.plot(time_plot, np.rad2deg(joint_lh_positions[:, 1]))
    plt.plot(time_plot, np.rad2deg(joint_rh_positions[:, 1]))
    plt.ylabel('Knee Angle [deg]')
    plt.legend(['Left','Right'],loc='upper right')
    plt.grid('on')
    plt.subplot(3,1,3)
    plt.plot(time_plot, np.rad2deg(joint_lh_positions[:, 2]))
    plt.plot(time_plot, np.rad2deg(joint_rh_positions[:, 2]))
    plt.grid('on')
    plt.ylabel('Ankle Angle [deg]')
    plt.legend(['Left','Right'],loc='upper right')
    plt.xlabel('Time [s]')

    return

def plot_gait_footsteps(t_start,t_stop):
     [time,
     ankle_l_trajectory,
     ankle_r_trajectory,
     foot_l_contact,
     foot_r_contact,
     muscle_lh_activations,
     muscle_rh_activations,
     muscle_lh_forces,
     muscle_rh_forces,
     joint_lh_positions,
     joint_rh_positions] = load_data()

     index_start = np.where(time == t_start)[0][0]
     index_end = np.where(time == t_stop)[0][0]
    
     time_plot = time[index_start:index_end+1]
     foot_r_contact = foot_r_contact[index_start:index_end+1,:]
     foot_l_contact = foot_l_contact[index_start:index_end+1,:]
    
     contact_data = np.hstack((foot_r_contact, foot_l_contact))
     plot_gait(time_plot, contact_data,  0.01)
     plt.show()
     return

if __name__ == '__main__':
    plt.close("all")
    t_start = 6
    t_stop = 8
    plot_joint_angles(t_start,t_stop)
    #plot_gait_footsteps(t_start,t_stop)
    plot_trajectories_XYZ(t_start,t_stop)
    plot_muscle_activations('right',t_start,t_stop)
