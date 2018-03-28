""" Force-Velocity Setup """
import numpy as np
import biolog


def mass_equation(pos, vel, force, mass_params):
    """ Mass equation"""
    #returns acceleration of mass
    g=mass_params.g
    load=mass_params.mass
    #biolog.warning("Implement the mass and muscle equation")
    return -g+force/load


def mass_system(pos, vel, force, mass_params):
    """ Muscle-Mass System"""
    #returns velocity and acceleration of mass
    return np.array([vel,mass_equation(pos, vel, force, mass_params)])  

