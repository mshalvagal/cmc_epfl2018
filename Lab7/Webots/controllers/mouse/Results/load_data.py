""" Load results data """

import numpy as np


def load_data():
    """ Load results data """
    ankle_l_trajectory = np.load("ankle_l_trajectory.npy")
    ankle_r_trajectory = np.load("ankle_r_trajectory.npy")
    return [ankle_l_trajectory, ankle_r_trajectory]


def main():
    """ Main """
    ankle_l_trajectory, ankle_r_trajectory = load_data()
    print("ankle_l_trajectory:\n{}\nankle_r_trajectory:\n{}".format(
        ankle_l_trajectory,
        ankle_r_trajectory
    ))
    return


if __name__ == '__main__':
    main()
