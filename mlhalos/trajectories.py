"""
:mod:`trajectories`

Computes trajectories of set of particles given initial conditions parameters.
"""

import numpy as np

from . import density
from . import parameters
from . import window


class Trajectories(object):
    """Obtains EPS trajectories for in and out particles."""

    def __init__(self, init_parameters, particles='all',
                 num_of_filters=50, num_particles=0):
        """
        Return trajectories of particles given initial conditions parameters.

        Args:
            init_parameters (class): instance :class:`InitialConditionsParameters`.
            particles (str, list): 'all'- all in and out particles,
                'random' - number of random in and out particles specified in num_particles,
                or a valid list of particle ids.
            num_of_filters (int): Number of top-hat smoothing filters to apply to the density field.
            num_particles (int): if particles='random', number of random in and out particles

        Returns:
            delta_in (SimArray): Density contrasts of in particles for n smoothing scales,
                of form [n_deltas,m_particles].
            delta_out (SimArray): Density contrasts of out particles for n smoothing scales,
                of form [n_deltas,m_particles].
        """

        window_parameters = window.WindowParameters(initial_parameters=init_parameters,
                                                    num_filtering_scales=num_of_filters)
        density_contrasts = density.DensityContrasts(initial_parameters=init_parameters,
                                                     num_filtering_scales=num_of_filters)

        if particles == "all":
            self.generate_all_trajectories(density_contrasts)

        elif particles == "random":
            self.generate_random_trajectories(density_contrasts, num_particles)

        elif isinstance(particles, (np.ndarray, list)):
            self.generate_trajectories_from_list_particles(particles, density_contrasts,
                                                           init_parameters, num_of_filters)
        else:
            raise TypeError("Invalid type of particles. Particles must be either all, random or a given list.")

        self.mass_spheres = window_parameters.smoothing_masses

    def generate_all_trajectories(self, density_contrasts):
        self.delta_in = density_contrasts.density_contrast_in
        self.delta_out = density_contrasts.density_contrast_out

    def generate_random_trajectories(self, density_contrasts, num_particles):
        self.delta_in = density_contrasts.density_contrast_in[np.random.choice(len(
            density_contrasts.density_contrast_in), num_particles)]
        self.delta_out = density_contrasts.density_contrast_out[np.random.choice(len(
            density_contrasts.density_contrast_out), num_particles)]

    def generate_trajectories_from_list_particles(self, particles, density_contrasts, init_parameters, num_of_filters):
        particles_in = [x for x in particles if x in init_parameters.ids_IN]
        particles_out = [x for x in particles if x in init_parameters.ids_OUT]

        d = density_contrasts
        self.delta_in = d.get_density_contrasts_subset_particles(init_parameters, num_of_filters,
                                                                 ids_type=particles_in)
        self.delta_out = d.get_density_contrasts_subset_particles(init_parameters, num_of_filters,
                                                                  ids_type=particles_out)


def get_num_particles_that_never_cross_threshold(density_contrast):
    """Get number of particles whose trajectories never cross density threshold."""
    output2 = []

    for i in range(len(density_contrast)):
        if any(num >= 1.0169 for num in density_contrast[i]):
            output2.append(i)

    output = list(set(range(len(density_contrast))) - set(output2))
    return len(output)


def get_index_particles_that_never_cross_threshold(density_contrast):
    """Get index of particles whose trajectories never cross density threshold."""
    crossing_index = []

    for i in range(len(density_contrast)):
        if any(num >= 1.0169 for num in density_contrast[i]):
            crossing_index.append(i)

    not_crossing_index = [j for j in range(len(density_contrast)) if j not in crossing_index]

    return not_crossing_index


def get_num_particles_that_cross_the_threshold(delta_out):
    """Get number of particles whose trajectories cross density threshold at least once."""
    output2 = []

    for i in range(len(delta_out)):
        if any(num >= 1.0169 for num in delta_out[i]):
            output2.append(i)

    return len(output2)


def get_index_particles_that_cross_the_threshold(delta_out):
    """Get index of particles whose trajectories cross density threshold at least once."""
    crossing_index = []

    for i in range(len(delta_out)):
        if any(num >= 1.0169 for num in delta_out[i]):
            crossing_index.append(i)

    return crossing_index
