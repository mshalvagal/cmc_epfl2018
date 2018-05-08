""" Plot gait """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def y_options(**kwargs):
    """ Return y options """
    y_size = kwargs.pop("y_size", 4)
    y_sep = 1.0/(y_size+1)

    def y_pos(y): return (y+(y+1)*y_sep)/(y_size+1)

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
    y_axis = [y_pos(y)+0.5*y_sep for y in range(4)]
    plt.yticks(y_axis, y_values)
    plt.xlabel("Time [s]")
    plt.ylabel("Gait")
    plt.axis('auto')
    plt.grid(True)
    return


def main():
    """ Main """
    dt = 0.001
    time = np.arange(0, 1, dt)
    gait = np.array(2*np.random.ranf([len(time), 4]), dtype=np.int)
    print(gait)
    plot_gait(time, gait[:, :2], dt, figurename="gait2")
    plt.show()
    return


if __name__ == '__main__':
    main()
