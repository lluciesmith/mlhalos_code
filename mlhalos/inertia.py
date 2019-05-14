"""
:mod:`inertia`

Computes the inertia tensor at each particle's position w.r.t nearest density peak.
Density peaks are local maxima of the Gaussian random field.

"""
import numpy as np
from mlhalos import parameters
from mlhalos import density
from mlhalos import shear
from mlhalos import window
from sklearn.neighbors import NearestNeighbors
import pynbody
import scipy.linalg.lapack as la


class DensityPeaks:
    def __init__(self, initial_parameters=None, density_contrasts=None, num_filtering_scales=50, path=None,
                 window_function="top hat"):
        """
        Instantiates :class:`DensityPeaks` given the

        Args:

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        self.initial_parameters = initial_parameters
        self.delta_class = density.DensityContrasts(initial_parameters=initial_parameters,
                                                    num_filtering_scales=num_filtering_scales,
                                                    window_function=window_function)
        if density_contrasts is None:
            density_contrasts = self.delta_class.density_contrasts

        self.density_contrasts = density_contrasts
        assert self.density_contrasts.shape == (num_filtering_scales, initial_parameters.shape**3)

    # Function for a single smoothed density contrast field

    def get_positions_local_maxima_density_contrast(self, delta):
        """ Returns the positions of the `delta` local maxima """
        ids_maxima = self.delta_class.get_position_local_maxima(delta)
        return ids_maxima

    def get_particle_ids_local_maxima_density_contrast(self, delta):
        """ Returns the particle ids associated with the `delta` local maxima """
        ids_maxima = self.delta_class.get_particle_ids_local_maxima(delta)
        return ids_maxima

    def get_nearest_peak(self, particle_id_peaks, position_peaks, particles_positions=None, snapshot=None,
                         output="id+position"):
        if snapshot is None:
            snapshot = self.initial_parameters.initial_conditions
        if particles_positions is None:
            particles_positions = snapshot["pos"]

        extended_peaks = self.extended_peaks_with_periodic_boundary_conditions(particle_id_peaks, position_peaks,
                                                                               snapshot)
        peak_ids_indices = self.find_kNearestNeighbour_ml(particles_positions, extended_peaks[:, 1:])

        if output == "id+position":
            return extended_peaks[peak_ids_indices, :]
        elif output == "particle id":
            return extended_peaks[peak_ids_indices, 0]
        elif output == "position":
            return extended_peaks[peak_ids_indices, 1:]
        else:
            raise ValueError("Select either position or particle id or both as output.")

    # Extend box to account for periodicity

    def extended_peaks_with_periodic_boundary_conditions(self, particle_ids_peaks, position_peaks, snapshot):
        """ 1st column is the particle-ID of the peak. 2nd, 3rd, 4th are the x,y,z coordinates"""
        boxsize = self.initial_parameters.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=snapshot, units="kpc")
        extra_length = boxsize/4

        pos_peaks_with_label = np.zeros((len(position_peaks), 4))
        pos_peaks_with_label[:, 0] = particle_ids_peaks
        pos_peaks_with_label[:, 1:] = position_peaks

        extra_peaks = self.pos_peaks_in_extended_box_of_periodic_bc(particle_ids_peaks, position_peaks, boxsize,
                                                                    extra_length)
        return np.vstack((pos_peaks_with_label, extra_peaks))

    def pos_peaks_in_extended_box_of_periodic_bc(self, particle_ids, positions, boxsize, extra_length):
        extra_right = boxsize - extra_length

        empty_array = []
        for i in range(len(positions)):
            ar = self.get_extra_peaks_in_extended_box_for_single_peak(positions[i], boxsize, extra_length, extra_right)
            if ar:
                ar_i = np.zeros((len(ar), 4))
                ar_i[:,0] = particle_ids[i].astype("int")
                ar_i[:, 1:] = ar
                empty_array.append(ar_i)

        return np.vstack(empty_array)

    @staticmethod
    def get_extra_peaks_in_extended_box_for_single_peak(peak_coordinates, boxsize, extra_length, extra_right):
        extra_pos = []

        x, y, z = peak_coordinates[0], peak_coordinates[1], peak_coordinates[2]

        if x < extra_length:
            extra_pos.append([x + boxsize, y, z])

            if y < extra_length:
                extra_pos.append([x, y + boxsize, z])
                extra_pos.append([x + boxsize, y + boxsize, z])

                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x, y + boxsize, z + boxsize])
                    extra_pos.append([x + boxsize, y, z + boxsize])
                    extra_pos.append([x + boxsize, y + boxsize, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x, y + boxsize, z - boxsize])
                    extra_pos.append([x + boxsize, y, z - boxsize])
                    extra_pos.append([x + boxsize, y + boxsize, z - boxsize])

                else:
                    pass

            elif y > extra_right:
                extra_pos.append([x, y - boxsize, z])
                extra_pos.append([x + boxsize, y - boxsize, z])

                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x, y - boxsize, z + boxsize])
                    extra_pos.append([x + boxsize, y, z + boxsize])
                    extra_pos.append([x + boxsize, y - boxsize, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x, y - boxsize, z - boxsize])
                    extra_pos.append([x + boxsize, y, z - boxsize])
                    extra_pos.append([x + boxsize, y - boxsize, z - boxsize])

                else:
                    pass

            else:
                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x + boxsize, y, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x + boxsize, y, z - boxsize])
                else:
                    pass

        elif x > extra_right:
            extra_pos.append([x - boxsize, y, z])

            if y < extra_length:
                extra_pos.append([x, y + boxsize, z])
                extra_pos.append([x - boxsize, y + boxsize, z])

                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x, y + boxsize, z + boxsize])
                    extra_pos.append([x - boxsize, y, z + boxsize])
                    extra_pos.append([x - boxsize, y + boxsize, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x, y + boxsize, z - boxsize])
                    extra_pos.append([x - boxsize, y, z - boxsize])
                    extra_pos.append([x - boxsize, y + boxsize, z - boxsize])
                else:
                    pass

            elif y > extra_right:
                extra_pos.append([x, y - boxsize, z])
                extra_pos.append([x - boxsize, y - boxsize, z])

                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x, y - boxsize, z + boxsize])
                    extra_pos.append([x - boxsize, y, z + boxsize])
                    extra_pos.append([x - boxsize, y - boxsize, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x, y - boxsize, z - boxsize])
                    extra_pos.append([x - boxsize, y, z - boxsize])
                    extra_pos.append([x - boxsize, y - boxsize, z - boxsize])
                else:
                    pass

            else:
                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x - boxsize, y, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x - boxsize, y, z - boxsize])
                else:
                    pass
        else:
            if y < extra_length:
                extra_pos.append([x, y + boxsize, z])

                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x, y + boxsize, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x, y + boxsize, z - boxsize])
                else:
                    pass

            elif y > extra_right:
                extra_pos.append([x, y - boxsize, z])

                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])
                    extra_pos.append([x, y - boxsize, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                    extra_pos.append([x, y - boxsize, z - boxsize])
                else:
                    pass

            else:
                if z < extra_length:
                    extra_pos.append([x, y, z + boxsize])

                elif z > extra_right:
                    extra_pos.append([x, y, z - boxsize])
                else:
                    pass

        return extra_pos

    # Use scikit-learn KNearestNeighbours algorithm

    def find_kNearestNeighbour_ml(self, particles_positions, peak_positions, return_distance=False):
        NN = NearestNeighbors(1, algorithm="kd_tree")
        NN.fit(peak_positions)

        nn = NN.kneighbors(particles_positions, 1, return_distance=return_distance)

        if return_distance is True:
            return nn[0].ravel(), nn[1].ravel()
        else:
            return nn.ravel()


class Inertia(DensityPeaks):
    """ Compute the inertia tensor (3x3 array) w.r.t. the nearest density peak for each particle id. """

    def __init__(self, initial_parameters=None, density_contrasts=None, num_filtering_scales=50, path=None,
                 window_function="top hat", number_of_processors=10):
        """
        Instantiates :class:`Inertia` given the

        Args:

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        self.initial_parameters = initial_parameters
        self.delta_class = density.DensityContrasts(initial_parameters=initial_parameters,
                                                    num_filtering_scales=num_filtering_scales,
                                                    window_function=window_function)
        self.shear_class = shear.Shear(initial_parameters=initial_parameters,
                                       num_filtering_scales=num_filtering_scales,
                                       number_of_processors=number_of_processors)

        DensityPeaks.__init__(self, initial_parameters=initial_parameters, density_contrasts=density_contrasts,
                              num_filtering_scales=num_filtering_scales, path=path, window_function=window_function)

    # Function for a single smoothed density contrast field

    def get_inertia_tensor_wrt_nearest_peak(self, density_contrast):
        dx, dy, dz = self.get_coordinates_distance_nearest_peak(density_contrast)
        inertia_tensor = self.compute_inertia_tensor(dx, dy, dz)
        return inertia_tensor

    def get_coordinates_distance_nearest_peak(self, density_contrast):
        position_particles = self.initial_parameters.initial_conditions["pos"]

        pos_peaks = self.get_positions_local_maxima_density_contrast(density_contrast)
        peak_id = self.get_particle_ids_local_maxima_density_contrast(density_contrast)

        nearest_peak = self.get_nearest_peak(peak_id, pos_peaks, output="position")
        dx, dy, dz = self.get_difference_coordinates(position_particles, nearest_peak)
        return dx, dy, dz

    @staticmethod
    def compute_inertia_tensor(x_coord, y_coord, z_coord, constant=None):
        I_ij = np.zeros((len(x_coord), 3, 3))

        for i in range(len(x_coord)):
            I_ij[i, 0] = [x_coord[i] * ijk for ijk in [x_coord[i], y_coord[i], z_coord[i]]]
            I_ij[i, 1] = [y_coord[i] * ijk for ijk in [x_coord[i], y_coord[i], z_coord[i]]]
            I_ij[i, 2] = [z_coord[i] * ijk for ijk in [x_coord[i], y_coord[i], z_coord[i]]]

            if constant is not None:
                I_ij[i] = constant[i] * I_ij[i]
        return I_ij

    @staticmethod
    def get_difference_coordinates(position_particles, position_nearest_peak):
        dl = position_particles - position_nearest_peak
        dx, dy, dz = dl[:, 0], dl[:, 1], dl[:, 2]
        return dx, dy, dz

    def get_eigenvalues_inertia_tensor(self, inertia_tensors, number_of_processors=10):
        eig = self.shear_class.get_eigenvalues_many_matrices(inertia_tensors, number_of_processors=number_of_processors)
        return eig

    def get_eigenvalues_inertia_tensor_from_density(self, density_contrast, number_of_processors=10):
        inertia_tensor = self.get_inertia_tensor_wrt_nearest_peak(density_contrast)
        eig_inertia = self.get_eigenvalues_inertia_tensor(inertia_tensor, number_of_processors=number_of_processors)
        return eig_inertia

    def get_eigenvalues_inertia_tensor_from_multiple_densities(self, densities, number_of_processors=10):
        shape = self.initial_parameters.shape

        eig_0 = np.zeros((shape ** 3, densities.shape[0]))
        eig_1 = np.zeros((shape ** 3, densities.shape[0]))
        eig_2 = np.zeros((shape ** 3, densities.shape[0]))

        for i in range(densities.shape[0]):
            print("Start loop " + str(i))
            d_i = densities[i]
            assert len(d_i) == shape**3

            eig_i = self.get_eigenvalues_inertia_tensor_from_density(d_i,  number_of_processors=number_of_processors)

            eig_0[:, i] = eig_i[:, 0]
            eig_1[:, i] = eig_i[:, 1]
            eig_2[:, i] = eig_i[:, 2]
            print("End loop " + str(i))

        return eig_0, eig_1, eig_2


class LocalInertia:
    """ Compute the inertia tensor (3x3 array) for each particle. """

    def __init__(self, ids_particles,  initial_parameters=None, num_filtering_scales=50, path=None):
        """
        Instantiates :class:`Inertia` given the

        Args:

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        self.initial_parameters = initial_parameters
        self.filter_parameters = window.WindowParameters(initial_parameters=initial_parameters,
                                                    num_filtering_scales=num_filtering_scales)
        # self.delta_class = density.DensityContrasts(initial_parameters=initial_parameters,
        #                                             num_filtering_scales=num_filtering_scales,
        #                                             window_function=window_function)
        # self.shear_class = shear.Shear(initial_parameters=initial_parameters,
        #                                num_filtering_scales=num_filtering_scales,
        #                                number_of_processors=number_of_processors)
        #
        # Inertia.__init__(self, initial_parameters=initial_parameters, density_contrasts=density_contrasts,
        #
        #                     num_filtering_scales=num_filtering_scales, path=path, window_function=window_function)

        self.filt_scales = np.arange(50)
        self.ids_particles = ids_particles
        # self.local_inertia(self.ids_particles, filt_scales)

    def get_local_inertia_single_id(self, particle_id, snapshot, smoothing_radii, const):
        tr = pynbody.analysis.halo.center(snapshot[particle_id], mode="hyb", vel=False, wrap=True)
        x, y, z = snapshot["x"], snapshot["y"], snapshot["z"]

        # pos_particle = (x[particle_id], y[particle_id], z[particle_id])
        # ids_vol = snapshot[pynbody.filt.Sphere(r_scale, cen=pos_particle)]["iord"]
        # x_diff, y_diff, z_diff = x[ids_vol] - x[particle_id], y[ids_vol] - y[particle_id], z[ids_vol] - z[particle_id]
        # i_ij = Inertia.compute_inertia_tensor(x_diff, y_diff, z_diff, constant=rho[ids_vol])

        i_ij_particle = np.zeros((len(smoothing_radii), 3, 3))
        eig_i = np.zeros((len(smoothing_radii), 3))

        for i in range(len(smoothing_radii)):
            smoothing_radius = smoothing_radii[i]

            ids_vol = snapshot[pynbody.filt.Sphere(smoothing_radius, cen=(0, 0, 0))]["iord"]
            i_ij = Inertia.compute_inertia_tensor(x[ids_vol], y[ids_vol], z[ids_vol], constant=const[ids_vol])
            i_ij_particle_i = np.sum(i_ij, axis=0)
            i_ij_particle[i] = i_ij_particle_i
            eig_real, eig_im, eigvec_real, eigvec_im, info = la.dgeev(i_ij_particle_i, compute_vl=0, compute_vr=0, overwrite_a=1)
            eig_i[i] = np.sort(eig_real)[::-1]

        return i_ij_particle, eig_i

    # def local_inertia_at_smoothing_scale(self, particle_ids, snapshot, r_scale, constant):
    #     inertia_all_particles = np.zeros((len(snapshot), 3, 3))
    #     for particle_id in particle_ids:
    #         inertia_all_particles[particle_id] = self.get_local_inertia_single_id(particle_id, snapshot, r_scale, constant)
    #
    #     return inertia_all_particles

    def local_inertia(self, particle_ids, filtering_scales, path="/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/"):
        snapshot = self.initial_parameters.initial_conditions
        rho = snapshot["rho"]
        # rho_mean = (np.sum(snapshot["mass"])/snapshot.properties["boxsize"]**3).in_units("Msol kpc**-3")
        # C = rho - rho_mean

        r_smoothing = self.filter_parameters.smoothing_radii.in_units(snapshot["pos"].units)[filtering_scales]

        for particle_id in particle_ids:
            print("Starting particle " + str(particle_id))

            i_ij_particle = self.get_local_inertia_single_id(particle_id, snapshot, r_smoothing, rho)
            np.save(path + "inertia_tensor_particle_" + str(particle_id) + ".npy", i_ij_particle)

            print("Done and saved particle " + str(particle_id))
            del i_ij_particle

