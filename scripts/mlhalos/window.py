"""
:mod:`window`

Computes top-hat window function in :class:`TopHat` and stores its radius and mass in
:class:`WindowParameters`, given instance :class:`InitialConditionsParameters`.

"""

# /Users/lls/Documents/CODE/stored_files
# /home/lls/stored_files

import pynbody
import numpy as np
import scipy.special

from . import parameters


class WindowParameters(object):
    """
    :class:`WindowParameters`

    Stores radius and mass of top-hat window functions.
    """

    def __init__(self, initial_parameters=None, num_filtering_scales=50, snapshot=None, volume="sphere"):
        """
        Instantiates :class:`WindowParameters` for given instance of
        :class:`InitialConditionsParameters` in :mod:`parameters`.

        Args:
            initial_parameters (class): Instance of :class:`InitialConditionsParameters`. Provides
                snapshot properties and mass range of in halos.
            num_filtering_scales (int): Number of top-hat smoothing filters to apply to the density field.
            snapshot (SimSnap): Optional choice of snapshot. Default is initial conditions snapshot.
            nyquist (float): This is the wavelength corresponding to the Nyquist frequency in Fourier space.
                We recommend choosing scales above the Nyquist.
            volume(str): This is how to define the volume of the filter function. Default ("sphere") is the volume of a
                sphere, "sharp-k" corresponds to V = 6*pi^2*R^3 which is a conventional choice in the literature.


        Returns:
            smoothing radii (SimArray): Top-hat filter radii of form [n_window_functions, ].
            smoothing masses (SimArray): Top-hat filter masses of form [n_window_functions, ].
        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters()

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        self.volume = volume

        self.smoothing_radii = self.list_smoothing_radius_from_initial_inputs(initial_parameters,
                                                                              num_filtering_scales, snapshot)
        self.smoothing_masses = self.array_smoothing_masses_log_spaced(initial_parameters, num_filtering_scales)

        # Print a warning if any of the smoothing scales are below the Nyquist.

        nyquist = initial_parameters.boxsize_no_units * 2 / initial_parameters.shape

        if (self.smoothing_radii < nyquist).any():
            print("WARNING: "
                  "Your choice of smoothing masses involves radii scales below the Nyquist frequency. This is not "
                  "recommended when using FFT. We recommend using smoothing mass scales larger than M = 3e10 Msol "
                  "for a top-hat in real space.")

    @staticmethod
    def array_smoothing_masses_log_spaced(initial_parameters, num_filtering_scales):
        """
        Returns masses of top-hat smoothing filter(SimArray).

        Given initial_parameters, instance of :class:`InitialConditionsParameters`, the smoothing masses
        are ``initial_parameters.num_of_filter`` log-spaced values in the range ``initial_parameters.M_min``
        and ``initial_parameters.M_max``.
        """
        mass = np.logspace(np.log10(initial_parameters.M_min), np.log10(initial_parameters.M_max),
                           num=num_filtering_scales, endpoint=True)

        mass_sim_array = pynbody.array.SimArray(np.array(mass))
        mass_sim_array.units = initial_parameters.M_max.units
        return mass_sim_array

    @staticmethod
    def get_radius_sphere_around_particle(initial_parameters, mass, mean_density, snapshot=None):
        """Calculates radius top-hat filter given its mass and mean matter density (SimArray)."""

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        radius = pynbody.array.SimArray(((4. * np.pi * mean_density)/(3. * mass))**(-1,3))\
            .in_units("Mpc", **snapshot.conversion_context())
        return radius

    def get_mass_from_radius(self, initial_parameters, radius, mean_density, snapshot=None):
        """Calculates mass top-hat filter given its mass and mean matter density (SimArray)."""

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        if self.volume == "sphere":
            factor = 4 * np.pi / 3
        elif self.volume == "sharp-k":
            factor = 6 * (np.pi ** 2)
        else:
            factor = 4 * np.pi / 3

        mass = pynbody.array.SimArray(factor * mean_density * (radius ** 3))
        mass.units = "Msol"
        return mass

    @staticmethod
    def get_radius_for_sharp_k_volume(initial_parameters, mass, mean_density, snapshot=None):
        """
        Calculates radius of a sharp-k filter given its mass and mean matter density (SimArray).
        The standard convention (also used here) is to assume V = 6 * pi^2 * R^3
        """

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        radius = pynbody.array.SimArray(((6. * (np.pi**2) * mean_density)/mass)**(-1,3))\
            .in_units("Mpc", **snapshot.conversion_context())
        return radius

    @staticmethod
    def get_radius_for_gaussian_volume(initial_parameters, mass, mean_density, snapshot=None):
        """
        Calculates radius of a sharp-k filter given its mass and mean matter density (SimArray).
        The standard convention (also used here) is to assume V = 6 * pi^2 * R^3
        """

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        V_gaussian = mass/mean_density
        radius = pynbody.array.SimArray((V_gaussian / ((2 * np.pi) ** (3 / 2))) ** (1, 3)).\
            in_units("Mpc", **snapshot.conversion_context())

        return radius

    def get_smoothing_radius_corresponding_to_filtering_mass(self, initial_parameters, mass_sphere, snapshot=None):
        """"Returns radius top-hat filter given its mass and initial_parameters (SimArray)."""

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        mean_matter_density = initial_parameters.get_mean_matter_density_in_the_box(snapshot=snapshot)
        # mean_matter_density = initial_parameters.mean_density

        volume = self.volume
        if volume == "sphere":
            radius = self.get_radius_sphere_around_particle(initial_parameters, mass_sphere, mean_matter_density,
                                                            snapshot=snapshot)
        elif volume == "sharp-k":
            radius = self.get_radius_for_sharp_k_volume(initial_parameters, mass_sphere, mean_matter_density,
                                                        snapshot=snapshot)
        elif volume == "gaussian":
            radius = self.get_radius_for_gaussian_volume(initial_parameters, mass_sphere, mean_matter_density,
                                                        snapshot=snapshot)
        else:
            # radius = []
            raise NameError("Select an appropriate form for the volume")
        return radius

    def list_smoothing_radius_from_initial_inputs(self, initial_parameters, num_filtering_scales, snapshot=None):
        """Returns list of top-hat radii given initial_parameters (SimArray)."""

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        mass = self.array_smoothing_masses_log_spaced(initial_parameters, num_filtering_scales)
        radius = self.get_smoothing_radius_corresponding_to_filtering_mass(initial_parameters, mass, snapshot=snapshot)

        return radius


class SharpK(object):

    def __init__(self, radius, initial_parameters=None, snapshot=None, path=None):
        """
        Instantiates :class:`SharpK` for given radius scale and instance of
        :class:`InitialConditionsParameters` in :mod:`parameters`.

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters()

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        # Need physical box size and number of grids in simulation box at snapshot to
        # compute top-hat filter.
        if path is None:
            self.path = initial_parameters.path

        self.boxsize = initial_parameters.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=snapshot)
        self.shape = int(scipy.special.cbrt(len(snapshot)))

        self.sharp_k_window = self.sharp_k(radius, self.boxsize, self.shape)

    def sharp_k(self, radius, boxsize, shape):
        if shape == 256:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix")
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix.npy')
                else:
                    print("loading top hat filter matrix from path " + str(self.path))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix.npy")

            except IOError:
                print("Calculating top hat filter matrix")
                a = TopHat.grid_coordinates_for_fourier_transform(shape)

        elif shape == 512:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_512.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_512.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = TopHat.grid_coordinates_for_fourier_transform(shape)

        elif shape == 2048:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_2048.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_2048.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = TopHat.grid_coordinates_for_fourier_transform(shape)

        else:
            a = TopHat.grid_coordinates_for_fourier_transform(shape)
        # Impose a[0, 0, 0] = 1 to avoid warnings for zero-division when evaluating top_hat.
        # Valid imposition since top_hat[0, 0, 0] need also to be 1.
        a[0, 0, 0] = 1

        k = 2.*np.pi*a/boxsize
        radius = float(radius)

        window = np.where(k <= (1/radius), 1, 0)
        return window

    @staticmethod
    def Wk(kR):
        return np.where((kR <= 1), 1, 0)


class Gaussian(object):

    def __init__(self, radius, initial_parameters=None, snapshot=None, path=None):
        """
        Instantiates :class:`SharpK` for given radius scale and instance of
        :class:`InitialConditionsParameters` in :mod:`parameters`.

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters()

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        # Need physical box size and number of grids in simulation box at snapshot to
        # compute top-hat filter.
        if path is None:
            self.path = initial_parameters.path

        self.boxsize = initial_parameters.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=snapshot)
        self.shape = int(scipy.special.cbrt(len(snapshot)))

        self.gaussian_window = self.gaussian_in_k_space(radius, self.boxsize, self.shape)

    def gaussian_in_k_space(self, radius, boxsize, shape):
        if shape == 256:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix")
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix.npy')
                else:
                    print("loading top hat filter matrix from path " + str(self.path))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix.npy")

            except IOError:
                print("Calculating top hat filter matrix")
                a = TopHat.grid_coordinates_for_fourier_transform(shape)
        elif shape == 512:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_512.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_512.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = TopHat.grid_coordinates_for_fourier_transform(shape)

        else:
            a = TopHat.grid_coordinates_for_fourier_transform(shape)
        # Impose a[0, 0, 0] = 1 to avoid warnings for zero-division when evaluating top_hat.
        # Valid imposition since top_hat[0, 0, 0] need also to be 1.
        a[0, 0, 0] = 1

        k = 2.*np.pi*a/boxsize
        radius = float(radius)

        window = np.exp(-((k**2)*(radius**2))/2)
        window[0,0,0] = 1

        return window

    @staticmethod
    def Wk(kR):
        return np.exp(-(kR**2)/2)


class TopHat(object):
    """
    :class:`TopHat`

    Defines top-hat window function in simulation box at given radius scale.
    """

    def __init__(self, radius, initial_parameters=None, snapshot=None, path=None):
        """
        Instantiates :class:`TopHat` for given radius scale and instance of
        :class:`InitialConditionsParameters` in :mod:`parameters`.

        Args:
            radius (SimArray): Radius of top-hat window function.
            initial_parameters (class): Instance of :class:`InitialConditionsParameters`. Provides
                snapshot properties and mass range of 'in' halos.
            snapshot (SimSnap): optional choice of snapshot. Default is initial conditions snapshot.

        Returns:
            top_hat_k_space (ndarray): Top-hat window function in Fourier space.

        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters()

        if snapshot is None:
            snapshot = initial_parameters.initial_conditions

        # Need physical box size and number of grids in simulation box at snapshot to
        # compute top-hat filter.
        if path is None:
            self.path = initial_parameters.path
        else:
            self.path = path

        self.boxsize = initial_parameters.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=snapshot)
        self.shape = int(scipy.special.cbrt(len(snapshot)))

        self.top_hat_k_space = self.top_hat_filter_in_k_space(radius, self.boxsize, self.shape)

    def top_hat_filter_in_k_space(self, radius, boxsize, shape):
        """
        Defines Fourier top-hat filter function in simulation box (ndarray).

        Args:
            radius (SimArray): Radius of top-hat window function.
            boxsize (array): physical size of simulation box.
            shape (int): number of grids in box.

        Returns:
            top_hat_k_space (ndarray): Top-hat window function in Fourier space.

        """

        if shape == 256:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    # print("loading top hat filter matrix")
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix.npy')
                else:
                    print("loading top hat filter matrix")
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix.npy")

            except IOError:
                print("Calculating top hat filter matrix")
                a = self.grid_coordinates_for_fourier_transform(shape)

        elif shape == 512:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_512.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_512.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = TopHat.grid_coordinates_for_fourier_transform(shape)

        elif shape == 2048:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_2048.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_2048.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = TopHat.grid_coordinates_for_fourier_transform(shape)

        else:
            a = self.grid_coordinates_for_fourier_transform(shape)
        # Impose a[0, 0, 0] = 1 to avoid warnings for zero-division when evaluating top_hat.
        # Valid imposition since top_hat[0, 0, 0] need also to be 1.
        a[0, 0, 0] = 1

        k = 2.*np.pi*a/boxsize
        radius = float(radius)

        top_hat = (3.*(np.sin(k*radius)-((k*radius)*np.cos(k*radius))))/((k*radius)**3)

        # we impose top_hat[0,0,0] = 1 since we want k=0 mode to be 1.
        top_hat[0, 0, 0] = 1.

        return top_hat

    @staticmethod
    def grid_coordinates_for_fourier_transform(shape):
        """Assigns coordinates to box grids (ndarray)."""

        a = np.zeros((shape, shape, shape))

        for i in range(shape):
            for j in range(shape):
                for k in range(shape):

                    if (i >= shape/2) and (j >= shape/2) and (k >= shape/2):
                        a[i, j, k] = np.sqrt((i-shape)**2 + (j-shape)**2 + (k-shape)**2)

                    elif (i >= shape/2) and (j >= shape/2) and(k < shape/2):
                        a[i, j, k] = np.sqrt((i-shape)**2 + (j-shape)**2 + k**2)

                    elif (i >= shape/2) and (j < shape/2) and (k >= shape/2):
                        a[i, j, k] = np.sqrt((i-shape)**2 + j**2 + (k-shape)**2)

                    elif (i < shape/2) and (j >= shape/2) and (k >= shape/2):
                        a[i, j, k] = np.sqrt(i**2 + (j-shape)**2 + (k-shape)**2)

                    elif (i >= shape/2) and (j < shape/2) and (k < shape/2):
                        a[i, j, k] = np.sqrt((i-shape)**2 + j**2 + k**2)

                    elif (i < shape/2) and (j >= shape/2) and (k < shape/2):
                        a[i, j, k] = np.sqrt(i**2 + (j-shape)**2 + k**2)

                    elif (i < shape/2) and (j < shape/2) and (k >= shape/2):
                        a[i, j, k] = np.sqrt(i**2 + j**2 + (k-shape)**2)

                    else:
                        a[i, j, k] = np.sqrt(i**2 + j**2 + k**2)
        return a

    @staticmethod
    def Wk(kR):
        return (3.*(np.sin(kR)-((kR)*np.cos(kR))))/((kR)**3)
