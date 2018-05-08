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


def main():
    """ Main """
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

    # Example to plot joint trajectories.
    # Feel free to change or use your own plot tools
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time, np.rad2deg(joint_lh_positions[:, 0]))
    plt.plot(time, np.rad2deg(joint_rh_positions[:, 0]))
    plt.ylabel('Angle [deg]')
    plt.grid('on')
    plt.subplot(3,1,2)
    plt.plot(time, np.rad2deg(joint_lh_positions[:, 1]))
    plt.plot(time, np.rad2deg(joint_rh_positions[:, 1]))
    plt.ylabel('Angle [deg]')
    plt.grid('on')
    plt.subplot(3,1,3)
    plt.plot(time, np.rad2deg(joint_lh_positions[:, 2]))
    plt.plot(time, np.rad2deg(joint_rh_positions[:, 2]))
    plt.grid('on')
    plt.ylabel('Angle [deg]')
    plt.xlabel('Time [s]')

    # Plot the ground contact of gait cycle
    contact_data = np.hstack((foot_r_contact, foot_l_contact))
    plot_gait(time, contact_data,  0.01)
    plt.show()

    return


if __name__ == '__main__':
    main()
