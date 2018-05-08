""" This file contains the class for muscle visualization in webots. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


class MuscleVisualization(object):
    """ Muscle visualization in webots """

    def __init__(self, muscle_attachment,
                 gps,
                 viz_transform,
                 viz_color,
                 viz_geom):

        super(MuscleVisualization, self).__init__()

        self.show_muscles = True
        self.transparency = 0

        self.muscle_attachment = muscle_attachment
        self.gps = gps
        self.viz_transform = viz_transform
        self.viz_color = viz_color
        self.viz_geom = viz_geom

        self.sides = ['LH', 'RH']

        # Muscle default vector
        self.muscle_vector_default = np.array([0., 1., 0.])

        # Color map
        self.color_map = None
        self.initialize_color_map()

    def compute_muscle_vector(self, gps_origin, gps_insertion, norm=False):
        """Compute the vector joining the origin and insertion of muscle."""

        vector = gps_origin - gps_insertion
        if norm:
            return (vector) / np.linalg.norm(vector)
        elif not norm:
            return vector

    def compute_muscle_length(self, muscle_vector):
        """Compute the length of the muscle vector."""
        return np.linalg.norm(muscle_vector)

    def compute_rotation_vector(self, muscle_vector, norm=False):
        """Compute the rotation vector of the default muscle vector.
        Function assumes that the muscle vector is normalized

        Parameters
        ----------
        self: type
            description
        muscle_vector: np.array
            Normalized muscle vector
        """

        vector = np.cross(self.muscle_vector_default,
                          muscle_vector)
        if norm:
            return vector / np.linalg.norm(vector)
        elif not norm:
            return vector

    def compute_rotation_angle(self, muscle_vector):
        """ Compute the rotation angle for the vector.

        Parameters
        ----------
        rotation_vector: np.array
            Rotation axis vector
        """

        return np.arccos(np.dot(self.muscle_vector_default,
                                muscle_vector) / np.linalg.norm(muscle_vector))

    def compute_center(self, gps_origin, gps_insertion):
        """Compute the center of a vector."""
        return (gps_origin + gps_insertion) / 2.

    def initialize_color_map(self):
        """Initialize color map for muscles."""
        reds = plt.get_cmap('Reds')
        cnorm = colors.Normalize(vmin=0, vmax=1)
        self.color_map = cmx.ScalarMappable(norm=cnorm, cmap=reds)

    def get_muscle_activation_color(self, activation):
        """Get the muscle activation color map."""
        color_ = self.color_map.to_rgba(activation)
        return list(color_[:3])

    def compute_muscle_vector_properties(self, gps_origin, gps_insertion):

        # Get muscle vector
        m_vector = self.compute_muscle_vector(
            gps_origin, gps_insertion, norm=True)

        # Get muscle vector length
        m_length = self.compute_muscle_length(self.compute_muscle_vector(
            gps_origin, gps_insertion))

        # Get muscle center
        m_center = self.compute_center(gps_origin, gps_insertion)

        # Get rotation_vector
        m_rotation_vector = self.compute_rotation_vector(m_vector, norm=True)

        # Get rotation angle
        m_rotation_angle = self.compute_rotation_angle(m_vector)

        return m_length, m_center, m_rotation_vector, m_rotation_angle

    def step(self, activations=None, viz=True):

        if viz:
            self.transparency = 0
        else:
            self.transparency = 1

        if viz:
            for side in self.sides:
                for muscle, attachments in self.muscle_attachment.items():

                    # if muscle == 'SOL':

                    # Compute muscle vector center, length, rotation axis and
                    # angle

                    gps_origin = side + '_' + attachments[0]
                    gps_insertion = side + '_' + attachments[1]

                    length, center, rotation_vector, rotation_angle = \
                        self.compute_muscle_vector_properties(
                            np.array(self.gps[gps_origin].getValues()),
                            np.array(self.gps[gps_insertion].getValues()))
                    # Transform
                    self.viz_transform[
                        side + '_MV_TRANSFORM_' + muscle
                    ].getField('translation').setSFVec3f(list(center))
                    # Rotation
                    rotation = list(rotation_vector)
                    rotation.append(rotation_angle)
                    self.viz_transform[
                        side + '_MV_TRANSFORM_' + muscle
                    ].getField('rotation').setSFRotation(rotation)
                    # Length
                    self.viz_geom[side + '_MV_GEOM_' + muscle].getField(
                        'height').setSFFloat(
                        length)
                    # Activation
                    act = activations[side + '_M_' + muscle]
                    self.viz_color[side + '_MV_COLOR_' + muscle].getField(
                        'diffuseColor').setSFColor(
                            self.get_muscle_activation_color(act))
                    self.viz_color[side + '_MV_COLOR_' + muscle].getField(
                        'transparency').setSFFloat(self.transparency)
            else:
                self.viz_color[side + '_MV_COLOR_' + muscle].getField(
                    'transparency').setSFFloat(self.transparency)
