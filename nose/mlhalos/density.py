"""
:mod:`density`

Computes density and density contrast for particles using top-hat smoothing.

"""
import numpy as np
import pynbody
import scipy.special
import scipy.signal
from multiprocessing import Pool
import pandas as pd

from . import window
from . import parameters


class Density(object):
    """ Computes densities for in and out particles"""

    def __init__(self, initial_parameters=None, num_filtering_scales=50, window_function="top hat", volume="sphere",
                 snapshot=None, shear_scale=None, path=None):
        """
        Instantiates :class:`density` using window function parameters
        from :class:`WindowParameters` in :mod:`window`.

        Args:
            initial_parameters (class): instance :class:`InitialConditionsParameters`.
            num_filtering_scales (int): Number of top-hat smoothing filters to apply
                to the density field.
            snapshot(str, None) : The snapshot at which to evaluate the density. If None,
                we use the initial conditions as defined in `initial_parameters`.
            shear_scale(int, None): The filtering scale at which to evaluate the shear field.
                Must be a value between (, num_filtering_scale).

        Returns:
            density_in (SimArray): Densities of in particles for n smoothing scales,
                of form [n_densities,m_particles].
            density_out (SimArray): Densities of out particles for n smoothing scales,
                of form [n_densities,m_particles].

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        self.initial_parameters = initial_parameters
        self.num_filtering_scales = num_filtering_scales
        self.snapshot = snapshot
        self.shear_scale = shear_scale
        self.path=path

        self.window_function = window_function
        self.volume = volume

        self._density_ = None
        self._filter_parameters_ = None
        self._mean_density_ = None

        self._density_in = None
        self._density_out = None
        self._smoothed_fourier_density = None

    @property
    def density_in(self):
        if self._density_in is None:
            self._density_in = self.get_subset_smooth_density(self.initial_parameters, self._density, self.snapshot,
                                                              ids_type="in")

        return self._density_in

    @property
    def density_out(self):
        if self._density_out is None:
            self._density_out = self.get_subset_smooth_density(self.initial_parameters, self._density, self.snapshot,
                                                               ids_type="out")
        return self._density_out

    @property
    def smoothed_fourier_density(self):
        if self._smoothed_fourier_density is None:

            if isinstance(self.shear_scale, int):
                radius_scale = self._filter_parameters.smoothing_radii[self.shear_scale]
                self._smoothed_fourier_density = self.get_smooth_density_k_space(self.initial_parameters, radius_scale,
                                                                                 self.snapshot, path=self.path)
            elif isinstance(self.shear_scale, (range, list, np.ndarray)):
                radius_scale = [self._filter_parameters.smoothing_radii[i] for i in self.shear_scale]
                self._smoothed_fourier_density = np.array([self.get_smooth_density_k_space(self.initial_parameters, j,
                                                                                           self.snapshot,
                                                                                           path=self.path)
                                                           for j in radius_scale])
        return self._smoothed_fourier_density

    @property
    def _density(self):
        if self._density_ is None:
            self._density_ = self.get_smooth_density_for_radii_list(self.initial_parameters,
                                                                    self._filter_parameters.smoothing_radii,
                                                                    self.snapshot, path=self.path)
        return self._density_

    @property
    def _filter_parameters(self):
        if self._filter_parameters_ is None:
            self._filter_parameters_ = window.WindowParameters(self.initial_parameters, self.num_filtering_scales,
                                                               snapshot=self.snapshot, volume=self.volume)

        return self._filter_parameters_

    @property
    def _mean_density(self):
        if self._mean_density_ is None:
            den = self._density
            mean_density = np.mean(den, axis=0)

            assert mean_density.shape[0] == self.num_filtering_scales
            self._mean_density_ = mean_density

        return self._mean_density_

    @staticmethod
    def get_density_in_box(parameter, snapshot=None):
        """
        Given snapshot reshape density from 1D array to 3D array
        corresponding to simulation box (SimArray).
        """
        if snapshot is None:
            snapshot = parameter.initial_conditions

        # Obtain density of particles as a pynbody SimArray in physical units.
        density = snapshot.dm['rho'].in_units('Msol Mpc**-3')
        shape = int(scipy.special.cbrt(len(snapshot)))

        density_reshaped = density.reshape((shape, shape, shape))
        return density_reshaped

    def get_density_k_space(self, parameter, snapshot=None):
        """Transform density to Fourier space (SimArray)."""

        if snapshot is None:
            snapshot = parameter.initial_conditions

        density = self.get_density_in_box(parameter, snapshot=snapshot)
        density_k_space = np.fft.fftn(density)

        return density_k_space

    def get_smooth_density_k_space(self, parameter, radius, snapshot=None, path=None):
        """Smooth density using a specified window function (SimArray)."""

        if snapshot is None:
            snapshot = parameter.initial_conditions

        density_k_space = self.get_density_k_space(parameter, snapshot=snapshot)

        window_function = self.window_function
        if window_function == "top hat":
            top_hat = window.TopHat(radius, initial_parameters=parameter, snapshot=snapshot, path=path)
            window_function = top_hat.top_hat_k_space

        elif window_function == "sharp k":
            sharp_k = window.SharpK(radius, initial_parameters=parameter, snapshot=snapshot, path=path)
            window_function = sharp_k.sharp_k_window

        elif window_function == "gaussian":
            gauss = window.Gaussian(radius, initial_parameters=parameter, snapshot=snapshot, path=path)
            window_function = gauss.gaussian_window

        den_smooth = window_function * density_k_space
        return den_smooth

    def get_smooth_density_real_space(self, parameter, radius, snapshot=None, path=None):
        """
        Transform smoothed density back to real space
        and reshape to 1D array (SimArray).
        """
        if snapshot is None:
            snapshot = parameter.initial_conditions

        den_smooth = self.get_smooth_density_k_space(parameter, radius, snapshot=snapshot, path=path)

        den_smooth_real = np.real(np.fft.ifftn(den_smooth).reshape((len(snapshot),)))
        return den_smooth_real

    def get_smooth_density_for_radii_list(self, parameter, r_list, snapshot=None, path=None):
        """
        Find density values for each radius for given particle (SimArray).
        Returns density in form [n_particles, m_densities]
        """

        if snapshot is None:
            snapshot = parameter.initial_conditions

        density_for_all_radii = pynbody.array.SimArray([self.get_smooth_density_real_space(parameter, r,
                                                                                           snapshot=snapshot, path=path)
                                                        for r in r_list])
        density_for_all_radii.units = self.get_density_in_box(parameter, snapshot=snapshot).units

        return density_for_all_radii.transpose()

    @staticmethod
    def get_subset_smooth_density(parameter, density, snapshot=None, ids_type='in'):
        """
        Compute density for each radius for particles of ids_type.

        Args:
            parameter (class): instance of :class:`InitialConditionParameters`
            density (SimArray): Density of all particles smoothed using top-hat filter.
            snapshot (SimSnap): optional variable for initial snapshot.
            ids_type (str): list of particle id numbers. Default is 'in' particles.

        Returns:
            density_of_ids (ndarray): Densities of particles for n smoothing functions,
                of form [n_particles, m_densities].

        Raises:
            TypeError: "Invalid subset of ids" if unknown ids_type is given.

        """
        if snapshot is None:
            snapshot = parameter.initial_conditions

        if ids_type == 'in':
            ids = parameter.ids_IN
        elif ids_type == 'out':
            ids = parameter.ids_OUT
        elif isinstance(ids_type, (np.ndarray, list)):
            ids = ids_type
        else:
            raise TypeError("Invalid subset of ids")

        density_of_ids = density[ids, :]
        return density_of_ids


class DensityContrasts(Density):
    """
    Computes density contrasts of in and out particles.

    This is a subclass of :class:`Density`.
    We define density contrast = density / mean matter density.
    """

    def __init__(self, initial_parameters=None, snapshot=None,
                 num_filtering_scales=50, window_function="top hat", volume="sphere", path=None):
        """
        Instantiates :class:`DensityContrasts`.

        Args:
            initial_parameters (class): instance :class:`InitialConditionsParameters`
            num_filtering_scales (int): Number of top-hat smoothing filters to apply
                to the density field.

        Returns:
            density_contrast_in (ndarray): Density contrast for in particles of form
                 [n_densities, m_particles].
            density_contrast_out (ndarray): Density contrast for out particles of form
                 [n_densities, m_particles].

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        self.initial_parameters = initial_parameters
        self.snapshot = snapshot

        Density.__init__(self, initial_parameters=initial_parameters, snapshot=snapshot,
                         num_filtering_scales=num_filtering_scales, path=path, window_function=window_function,
                         volume=volume)

        self._density_contrasts = None

        self._density_contrast_in = None
        self._density_contrast_out = None

        self._indices_local_maxima_ = None

    @property
    def density_contrasts(self):
        if self._density_contrasts is None:
            self._density_contrasts = self.get_density_contrast(self.initial_parameters, self._density)

        return self._density_contrasts

    @property
    def density_contrast_in(self):
        if self._density_contrast_in is None:
            self._density_contrast_in = self.get_density_contrast(self.initial_parameters, self.density_in)

        return self._density_contrast_in

    @property
    def density_contrast_out(self):
        if self._density_contrast_out is None:
            self._density_contrast_out = self.get_density_contrast(self.initial_parameters, self.density_out)

        return self._density_contrast_out

    def get_density_contrast(self, initial_parameter, density, rho_bar=None):
        """
        Computes density contrast given density and initial parameters.

        Args:
            rho_bar: Mean matter density in the Universe. Default is calculated in `Density`.
            initial_parameter (class): instance :class:`InitialConditionsParameters`
            density (SimArray): Density of particles.

        Returns:
            density_contrast (ndarray): density_contrast = density/mean_density.

        Raises:
            TypeError: Density and mean density must have same units.
        """
        if rho_bar is None:
            rho_bar = initial_parameter.get_mean_matter_density_in_the_box(snapshot=self.snapshot)

        #rho_bar = np.mean(density)

        if density.units == rho_bar.units:
            density_contrast = density / rho_bar
        else:
            raise TypeError("Density and mean density must have same units")
        return density_contrast

    def get_density_contrasts_subset_particles(self, initial_parameter, num_filtering_scales, ids_type='in'):
        """Computes density contrasts for particles in ids_type (ndarray)."""

        filter_parameters = window.WindowParameters(initial_parameters=initial_parameter,
                                                    num_filtering_scales=num_filtering_scales)
        density = self.get_smooth_density_for_radii_list(initial_parameter, filter_parameters.smoothing_radii)

        density_ids = self.get_subset_smooth_density(initial_parameter, density, ids_type=ids_type)
        delta_ids = self.get_density_contrast(initial_parameter, density_ids)

        return delta_ids

    ################# Local maxima of density contrast field #################

    # Positions local maxima

    def get_position_local_maxima(self, density_contrast, snapshot=None):
        """ Find indices of local maxima in the density_contrast field and then find corresponding positions in kpc."""

        if snapshot is None:
            snapshot = self.initial_parameters.initial_conditions

        max_indices = self.get_indices_local_maxima_in_3d_box(self.initial_parameters, density_contrast,
                                                              output="DataFrame")
        pos_max = self.get_position_from_index_maxima(max_indices, snapshot=snapshot)
        return pos_max

    def get_position_from_index_maxima(self, indices_maxima, snapshot=None):
        shape = int(self.initial_parameters.shape)

        pos_ids = snapshot["pos"].reshape(shape, shape, shape, 3)
        pos_max = pos_ids[indices_maxima["x"], indices_maxima["y"], indices_maxima["z"]]
        return pos_max

    # Particle ids local maxima

    def get_particle_ids_local_maxima(self, density_contrast):
        max_indices = self.get_indices_local_maxima_in_3d_box(self.initial_parameters, density_contrast,
                                                              output="DataFrame")
        ids_maxima = self.get_ids_from_index_maxima(max_indices)
        return ids_maxima

    def get_ids_from_index_maxima(self, indices_maxima):
        shape = int(self.initial_parameters.shape)

        particle_ids = np.arange(shape ** 3).reshape(shape, shape, shape)
        ids_maxima = particle_ids[indices_maxima["x"], indices_maxima["y"], indices_maxima["z"]]
        return ids_maxima

    # Indices of 3D box of local maxima

    @staticmethod
    def get_indices_local_maxima_in_3d_box(initial_parameters, density_contrast, output="DataFrame"):
        """ For shape=256, this takes t=1.27s to find ~10^3 maxima and t=2.5s to find ~7x10^4 maxima."""

        shape = int(initial_parameters.shape)
        assert density_contrast.shape == (shape**3, )
        d_reshaped = density_contrast.reshape(shape, shape, shape)

        x_i, x_j, x_k = scipy.signal.argrelmax(d_reshaped, axis=0, order=2, mode="wrap")
        y_i, y_j, y_k = scipy.signal.argrelmax(d_reshaped, axis=1, order=2, mode="wrap")
        z_i, z_j, z_k = scipy.signal.argrelmax(d_reshaped, axis=2, order=2, mode="wrap")

        df1 = pd.DataFrame({"x": x_i, "y": x_j, "z": x_k}).reset_index()
        df2 = pd.DataFrame({"x": y_i, "y": y_j, "z": y_k}).reset_index()
        df3 = pd.DataFrame({"x": z_i, "y": z_j, "z": z_k}).reset_index()
        maxima_field = pd.merge(pd.merge(df1, df2, left_on=["x", "y", "z"], right_on=["x", "y", "z"]),
                                df3, left_on=["x", "y", "z"], right_on=["x", "y", "z"])
        if output == "array":
            maxima_field = maxima_field.values[:, 1:4]

        return maxima_field




