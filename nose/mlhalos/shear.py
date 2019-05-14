"""
:mod:`shear`

Computes the shear tensor at radius scale `window.smoothing_radii[shear_scale]`.

The shear tensor field is a 3x3 matrix, given by the second derivative of the
gravitational potential. The gravitational potential is calculated by the Poisson
equation in Fourier space from the density field smoothed at a given radius
smoothing scale.

"""
import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import G
import scipy.linalg.lapack as la
from multiprocessing import Pool
import time

from . import window
from . import density
from . import parameters
# import importlib
# importlib.reload(window)
# importlib.reload(density)


class Shear(object):
    """ Compute the shear tensor (3x3 array) at each grid point of the box (shape x shape x shape)"""

    def __init__(self, initial_parameters=None, num_filtering_scales=50,
                 snapshot=None, shear_scale=27, number_of_processors=10, path=None):
        """
        Instantiates :class:`shear` given the density field smoothed at
        `window.smoothing_radii[shear_scale]` radius. 

        Args:
            initial_parameters:
            num_filtering_scales:
            snapshot:
            shear_scale:
        """
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        if snapshot is None:
            self.snapshot = initial_parameters.initial_conditions

        self.initial_parameters = initial_parameters
        self.num_filtering_scales = num_filtering_scales

        if shear_scale is "all":
            self.shear_scale = range(50)
        else:
            self.shear_scale = shear_scale

        self.number_of_processors = number_of_processors
        self.path = path

        self._density_scale_fourier_ = None
        self.boxsize = self.initial_parameters.boxsize_no_units
        self.shape = int(scipy.special.cbrt(len(self.snapshot)))

        self._shear_eigenvalues = None
        self._density_subtracted_eigenvalues = None

    @property
    def shear_eigenvalues(self):
        if self._shear_eigenvalues is None:
            self._shear_eigenvalues = self.calculate_shear_eigenvalues()

        return self._shear_eigenvalues

    @property
    def density_subtracted_eigenvalues(self):
        if self._density_subtracted_eigenvalues is None:
            self._density_subtracted_eigenvalues = self.calculate_density_subtracted_eigenvalues()

        return self._density_subtracted_eigenvalues

    @property
    def _density_scale_fourier(self):
        if self._density_scale_fourier_ is None:
            self._density_scale_fourier_ = self.get_smoothed_fourier_density(self.initial_parameters,
                                                                             self.num_filtering_scales, self.snapshot,
                                                                             self.shear_scale, path=self.path)
        return self._density_scale_fourier_

    @staticmethod
    def get_smoothed_fourier_density(initial_parameters, num_filtering_scales, snapshot, shear_scale, path=None):
        """
        Get the density of each particle in Fourier space smoothed at radius scale
        `w.smoothing_radii[shear_scale]` where w is an instantiaton of `window` class.
        """
        density_class = density.Density(initial_parameters=initial_parameters,
                                        num_filtering_scales=num_filtering_scales,
                                        shear_scale=shear_scale, snapshot=snapshot, path=path)
        density_scale = density_class.smoothed_fourier_density
        return density_scale

    def rescale_density_to_density_contrast_in_fourier_space(self, density_fourier):
        den_contrast_fourier = density_fourier / (density_fourier[0, 0, 0] / (self.shape ** 3))
        den_contrast_fourier[0, 0, 0] = 0
        return den_contrast_fourier

    def get_k_coordinates_box(self, shape, boxsize):
        """ Return the k-coordinates of the simulation box """

        if shape == 256:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix.npy')
                else:
                    print("Loading Fourier transform matrix for shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix.npy")

            except IOError:
                print("Calculating Fourier transform matrix for shape " + str(shape))
                a = window.TopHat.grid_coordinates_for_fourier_transform(shape)

        elif shape == 512:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_512.npy')
                    print("loading Fourier transform matrix of shape " + str(shape))
                else:
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_512.npy")
                    print("loading Fourier transform matrix of shape " + str(shape))

            except IOError:
                print("Calculating Fourier transform matrix for shape " + str(shape))
                a = window.TopHat.grid_coordinates_for_fourier_transform(shape)

        else:
            print("Not using FT matrix")
            a = window.TopHat.grid_coordinates_for_fourier_transform(shape)

        k_coord = 2. * np.pi * a / boxsize
        return k_coord

    def get_norm_k_box(self, shape, boxsize):
        k_coord = self.get_k_coordinates_box(shape, boxsize)
        norm_squared = np.power(np.fabs(k_coord), 2)
        return norm_squared

    @staticmethod
    def poisson_equation_in_fourier_space(rho_contrast_fourier, norm_k_squared):
        """ Rescaled Poisson's equation in Fourier space """
        norm_k_squared[0, 0, 0] = 1

        phi = - rho_contrast_fourier / norm_k_squared
        phi[0, 0, 0] = 0
        return phi

    # @staticmethod
    # def poisson_equation_in_fourier_space(rho_fourier, norm_k_squared):
    #     """ Poisson's equation in Fourier space """
    #     norm_k_squared[0, 0, 0] = 1
    #
    #     phi = - (4 * np.pi * G * rho_fourier) / norm_k_squared
    #     phi[0, 0, 0] = 0
    #     return phi

    def get_potential_from_density_fourier_space(self, density, shape, boxsize):
        """
        Computes the gravitational potential give the density via Poisson's equation in Fourier space.

        """
        den_contrast = self.rescale_density_to_density_contrast_in_fourier_space(density)
        k_norm_squared = self.get_norm_k_box(shape, boxsize)
        potential = self.poisson_equation_in_fourier_space(den_contrast, k_norm_squared)
        return potential

    def shear_prefactor_ki_kj(self, i, j, k, k_mode):
        nyquist = int(self.shape/2)
        k_mode_coordinates = np.zeros((3, 3))

        k_mode_coordinates[0] = [k_mode[i] * k_mode[ijk] for ijk in [i, j, k]]
        k_mode_coordinates[1] = [k_mode[j] * k_mode[ijk] for ijk in [i, j, k]]
        k_mode_coordinates[2] = [k_mode[k] * k_mode[ijk] for ijk in [i, j, k]]

        if any([mode == nyquist for mode in [i, j, k]]):

            if i == nyquist and j != nyquist and k!= nyquist:
                position_nyquist = 0
                position_not_nyquist = [1, 2]

            elif i != nyquist and j == nyquist and k != nyquist:
                position_nyquist = 1
                position_not_nyquist = [0, 2]

            elif i != nyquist and j != nyquist and k == nyquist:
                position_nyquist = 2
                position_not_nyquist = [0, 1]

            elif i == nyquist and j == nyquist and k != nyquist:
                position_nyquist = [0, 1]
                position_not_nyquist = 2

            elif i == nyquist and j != nyquist and k == nyquist:
                position_nyquist = [0, 2]
                position_not_nyquist = 1

            else:
                position_nyquist = [1, 2]
                position_not_nyquist = 0

            k_mode_coordinates[position_nyquist, position_not_nyquist] = 0
            k_mode_coordinates[position_not_nyquist, position_nyquist] = 0
        else:
            pass

        return k_mode_coordinates

    def calculate_shear_in_single_grid(self, potential, i, j, k, k_mode):
        k_mode_coordinates = self.shear_prefactor_ki_kj(i, j, k, k_mode)
        shear_single_grid = - k_mode_coordinates * potential[i, j, k]

        return shear_single_grid

    def get_shear_from_potential_in_fourier_space(self, potential, shape, boxsize):
        """
        Computes the 3x3 shear tensor at each grid point from the gravitational potential.
        This function takes ~5.25 minutes to run on a single processor.

        Args:
            subtract_density (bool):

        """
        shape = int(shape)
        a_space = np.concatenate((np.arange(shape/2), np.arange(-shape/2, 0)))
        k_mode = 2 * np.pi * a_space / boxsize

        shear_fourier = np.zeros((shape, shape, shape, 3, 3), dtype=complex)

        for i in range(shape):
            for j in range(shape):
                for k in range(shape):

                    if i == 0 and j == 0 and k == 0:
                        pass

                    else:
                        shear_ijk = self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
                        shear_fourier[i, j, k] = shear_ijk

        return shear_fourier

    def get_shear_tensor_in_real_space_from_potential(self, potential, shape, boxsize):
        """
        Returns the 3x3 shear tensor in real space at each grid point,
        after computing the sheat tensor from the potential in Fourier space.

        The computation takes ~5.25 minutes to run and the inverse
        Fourier transform takes ~26 seconds.

        """
        shear_fourier = self.get_shear_from_potential_in_fourier_space(potential, shape, boxsize)
        shear_real = np.real(np.fft.ifftn(shear_fourier, axes=(0, 1, 2)).reshape((shape**3, 3, 3)))

        return shear_real

    def get_shear_tensor_at_scale(self, density_scale):
        """
        Shear tensor computation.

        Args:
            density:
            shape:
            boxsize:

        Returns:

        """
        shape = self.shape
        boxsize = self.boxsize

        potential = self.get_potential_from_density_fourier_space(density_scale, shape, boxsize)
        shear_real = self.get_shear_tensor_in_real_space_from_potential(potential, shape, boxsize)

        return shear_real

    @staticmethod
    def get_eigenvalues_matrix(matrix):
        """
        Sort the eigenvalues such that eigval1 >= eigval2 >= eigval3.
        This function takes ~8 microseconds. It uses LAPACK Fortran package which is much faster
        than default scipy implementation (about ~10 times faster).
        """
        eig_real, eig_im, eigvec_real, eigvec_im, info = la.dgeev(matrix, compute_vl=0, compute_vr=0, overwrite_a=1)
        eig_sorted = np.sort(eig_real)[::-1]
        return eig_sorted

    def get_eigenvalues_many_matrices(self, matrices, number_of_processors=10):

        t00 = time.time()
        t0 = time.clock()

        if number_of_processors == 1:
            shear_eigenvalues = [self.get_eigenvalues_matrix(matrices[i]) for i in range(len(matrices))]

        else:
            pool = Pool(processes=number_of_processors)
            function = self.get_eigenvalues_matrix
            shear_eigenvalues = pool.map(function, matrices)
            pool.close()
            pool.join()

        print("Wall time " + str(time.time() - t00))
        print("Process time " + str(time.clock() - t0))

        return np.array(shear_eigenvalues)

    def get_eigenvalues_shear_tensor_from_density(self, density, number_of_processors=10):
        """
        Calculating the eigenvalues takes a long time.
        It is recommended to use multiprocessing - specify number of processors to use.

        """

        shear_tensor = self.get_shear_tensor_at_scale(density)
        shear_eigenvalues = self.get_eigenvalues_many_matrices(shear_tensor, number_of_processors=number_of_processors)
        return shear_eigenvalues

    def get_eigenvalues_shear_tensor_from_multiple_densities(self, densities, number_of_processors=10):
        shear_eigenvalues = np.zeros((self.shape**3, 3*densities.shape[0]))

        for i in range(densities.shape[0]):
            print("Start loop " + str(i))
            eig_i = self.get_eigenvalues_shear_tensor_from_density(densities[i],
                                                                   number_of_processors=number_of_processors)
            shear_eigenvalues[:, int(3 * i):int(3 * i) + 3] = eig_i
            print("End loop " + str(i))

        return shear_eigenvalues

    def calculate_shear_eigenvalues(self):

        density = self._density_scale_fourier
        print("Done densities fourier")
        number_of_processors = self.number_of_processors

        if len(density.shape) == 3:
            print("Calculating eigenvalues for single scale..")
            eigenvalues = self.get_eigenvalues_shear_tensor_from_density(density, number_of_processors)

        else:
            print("Calculating eigenvalues for multiple scales..")
            eigenvalues = self.get_eigenvalues_shear_tensor_from_multiple_densities(density, number_of_processors)

        return eigenvalues

    def get_sum_eigvals(self, eigvals):
        #eigvals = self.shear_eigenvalues

        if eigvals.shape[1] == 3:
            sum_eig = np.sum(eigvals, axis=1)

        else:
            sum_eig = np.column_stack([np.sum(eigvals[:, (3 * i): (3 * i) + 3], axis=1)
                                       for i in range(len(self.shear_scale))])
        return sum_eig

    @staticmethod
    def subtract_a_third_of_trace(eigenvalues_with_trace, sum_eigenvalues):
        density_to_subtract = np.column_stack((sum_eigenvalues / 3, sum_eigenvalues / 3, sum_eigenvalues / 3))
        density_subtracted_eigenvalues = eigenvalues_with_trace - density_to_subtract
        return density_subtracted_eigenvalues

    def subtract_density_from_eigenvalues(self, eigenvalues_with_trace, sum_eigenvalues):
        if eigenvalues_with_trace.shape[1] == 3:
            density_to_subtract = np.column_stack((sum_eigenvalues / 3, sum_eigenvalues / 3, sum_eigenvalues / 3))
            d_sub_eigenvalues = eigenvalues_with_trace - density_to_subtract

        else:
            d_sub_eigenvalues = np.zeros((eigenvalues_with_trace.shape))

            for i in range(len(self.shear_scale)):
                eig_i = eigenvalues_with_trace[:, int(3 * i):int(3 * i) + 3]
                sum_i = sum_eigenvalues[:, i]

                d_sub_eigenvalues[:, int(3 * i):int(3 * i) + 3] = self.subtract_a_third_of_trace(eig_i,  sum_i)

        return d_sub_eigenvalues

    def calculate_density_subtracted_eigenvalues(self):
        eigenvalues = self.shear_eigenvalues
        sum_eigenvalues = self.get_sum_eigvals(eigenvalues)

        density_subtracted_eigenvalues = self.subtract_density_from_eigenvalues(eigenvalues, sum_eigenvalues)
        return density_subtracted_eigenvalues


class ShearProperties(Shear):

    def __init__(self, initial_parameters=None, num_filtering_scales=50,
                 snapshot=None, shear_scale=27, number_of_processors=10, path=None):

        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path=path)

        Shear.__init__(self, initial_parameters=initial_parameters, num_filtering_scales=num_filtering_scales,
                       snapshot=snapshot, shear_scale=shear_scale, number_of_processors=number_of_processors, path=path)

        self._ellipticity = None
        self._prolateness = None

        self._density_subtracted_ellipticity = None
        self._density_subtracted_prolateness = None

        self._sum_eigvals = None

    @property
    def ellipticity(self):
        if self._ellipticity is None:
            self._ellipticity = self.get_ellipticity(subtract_density=False)

        return self._ellipticity

    @property
    def prolateness(self):
        if self._prolateness is None:
            self._prolateness = self.get_prolateness(subtract_density=False)

        return self._prolateness

    @property
    def density_subtracted_ellipticity(self):
        if self._density_subtracted_ellipticity is None:
            self._density_subtracted_ellipticity = self.get_ellipticity(subtract_density=True)

        return self._density_subtracted_ellipticity

    @property
    def density_subtracted_prolateness(self):
        if self._density_subtracted_prolateness is None:
            self._density_subtracted_prolateness = self.get_prolateness(subtract_density=True)

        return self._density_subtracted_prolateness

    def get_ellipticity(self, subtract_density=False):

        if isinstance(self.shear_scale, int):
            ellip = self.get_ellipticity_single_density_scale(subtract_density=subtract_density)

        elif isinstance(self.shear_scale, (range, list, np.ndarray)):
            ellip = self.get_ellipticity_for_multiple_density_scales(subtract_density=subtract_density)

        else:
            raise TypeError("The shear scale must be an integer or a list of integers")

        return ellip

    def get_prolateness(self, subtract_density=False):

        if isinstance(self.shear_scale, int):
            prol = self.get_prolateness_single_density_scale(subtract_density=subtract_density)

        elif isinstance(self.shear_scale, (range, list, np.ndarray)):
            prol = self.get_prolateness_for_multiple_density_scales(subtract_density=subtract_density)

        else:
            raise TypeError("The shear scale must be an integer or a list of integers")

        return prol

    def get_ellipticity_single_density_scale(self, subtract_density=False):
        # sort eigenvalues such that lambda_1 >= lambda_2 >= lambda_3

        if subtract_density is True:

            eigvals = self.density_subtracted_eigenvalues
            sum_ids = None

        else:
            eigvals = self.shear_eigenvalues
            sum_ids = self.get_sum_eigvals(eigvals)

        ellipticity = self.calculate_ellipticity(eigvals, sum_ids)
        return ellipticity

    def get_prolateness_single_density_scale(self, subtract_density=False):

        if subtract_density is True:

            eigvals = self.density_subtracted_eigenvalues
            sum_ids = None

        else:
            eigvals = self.shear_eigenvalues
            sum_ids = self.get_sum_eigvals(eigvals)

        prol = self.calculate_prolateness(eigvals, sum_ids)
        return prol

    def get_ellipticity_for_multiple_density_scales(self, subtract_density=False):
        ellipticity = np.zeros((self.shape**3, len(self.shear_scale)))

        if subtract_density is True:
            eigvals = self.density_subtracted_eigenvalues
            sum_eigvals = None
        else:
            eigvals = self.shear_eigenvalues
            sum_eigvals = self.get_sum_eigvals(eigvals)

        for i in range(len(self.shear_scale)):

            eig_i = eigvals[:, int(3*i): (3*i) + 3]

            if sum_eigvals is not None:
                sum_eigvals_i = sum_eigvals[:, i]
            else:
                sum_eigvals_i = None

            ellipticity[:, i] = self.calculate_ellipticity(eig_i, sum_eigvals_i)

        return ellipticity

    def get_prolateness_for_multiple_density_scales(self, subtract_density=False):
        prolateness = np.zeros((self.shape**3, len(self.shear_scale)))

        if subtract_density is True:
            eigvals = self.density_subtracted_eigenvalues
            sum_eigvals = None
        else:
            eigvals = self.shear_eigenvalues
            sum_eigvals = self.get_sum_eigvals(eigvals)

        for i in range(len(self.shear_scale)):

            eig_i = eigvals[:, int(3*i): (3*i) + 3]

            if sum_eigvals is not None:
                sum_eigvals_i = sum_eigvals[:, i]
            else:
                sum_eigvals_i = None

            prolateness[:, i] = self.calculate_prolateness(eig_i, sum_eigvals_i)

        return prolateness

    @staticmethod
    def calculate_ellipticity(eigenvalues, sum_ids=None):
        if sum_ids is None:
            ellipticity = (eigenvalues[:, 0] - eigenvalues[:, 2])

        else:
            ellipticity = (eigenvalues[:, 0] - eigenvalues[:, 2]) / (2 * sum_ids)

        return ellipticity

    @staticmethod
    def calculate_prolateness(eigenvalues, sum_ids=None):
        if sum_ids is None:
            prolateness = (3 * (eigenvalues[:, 0] + eigenvalues[:, 2]))

        else:
            prolateness = (eigenvalues[:, 0] + eigenvalues[:, 2] - (2 * (eigenvalues[:, 1]))) / (2 * sum_ids)

        return prolateness



def get_ids_type_eigenvalues(parameter, shear_property, ids_type='in', snapshot=None):

    if snapshot is None:
        snapshot = parameter.initial_conditions

    if ids_type == 'in':
        ids_type = parameter.ids_IN
    elif ids_type == 'out':
        ids_type = parameter.ids_OUT
    elif isinstance(ids_type, (np.ndarray, list)):
        ids_type = ids_type
    else:
        raise TypeError("Invalid subset of ids")

    shear_eigenvalues_of_ids = np.array([shear_property[particle_id] for particle_id in ids_type])
    return shear_eigenvalues_of_ids


def fractional_anisotropy(eigenvalues):

    norm = 1/np.sqrt(3)
    a02 = (eigenvalues[:, 0] - eigenvalues[:, 2])**2
    a01 = (eigenvalues[:, 0] - eigenvalues[:, 1]) ** 2
    a12 = (eigenvalues[:, 1] - eigenvalues[:, 2]) ** 2

    numerator = a02 + a01 + a12
    denominator = (eigenvalues[:, 0]**2) + (eigenvalues[:, 1]**2) + (eigenvalues[:, 2]**2)

    FA = norm * np.sqrt(numerator/denominator)
    return FA


def numerator_fa(eigenvalues):

    a02 = (eigenvalues[:, 0] - eigenvalues[:, 2])**2
    a01 = (eigenvalues[:, 0] - eigenvalues[:, 1]) ** 2
    a12 = (eigenvalues[:, 1] - eigenvalues[:, 2]) ** 2

    numerator = a02 + a01 + a12
    return np.sqrt(numerator)



# def calculate_density_factor_to_subtract(self, density):
#     density_real_space = np.real(np.fft.ifftn(density).reshape(self.shape**3,))
#     mean_density = np.mean(density_real_space)
#
#     subtract_trace = (density_real_space - mean_density)/(3*mean_density)
#     #subtract_trace_reshaped = np.tile(subtract_trace.reshape(self.shape**3, 1), 3)
#     return subtract_trace
# @staticmethod
# def get_eigenvalues_matrix(matrix):
#     """ Sort the eigenvalues such that eigval1 >= eigval2 >= eigval3"""
#     eig = scipy.linalg.eigvals(matrix)
#     eig_sorted = np.sort(eig)[::-1]
#     return eig_sorted
#
# def get_eigenvalues_shear_tensor(self, shear_tensor, number_of_processors=10):
#
#     if number_of_processors == 1:
#         shear_eigenvalues = np.array([self.get_eigenvalues_matrix(shear_tensor[i])
#                                             for i in range(len(shear_tensor))])
#
#     else:
#         pool = Pool(processes=number_of_processors)
#         function = self.get_eigenvalues_matrix
#         shear_eigenvalues = pool.map(function, shear_tensor)
#         pool.close()
#
#     return np.real(shear_eigenvalues)

# @property
# def shear_eigval_in(self):
#
#     if self._shear_eigval_in is None:
#         self._shear_eigval_in = self.generate_eigenvalues_type(subtract_density=False)
#
#     return self._shear_eigval_in
#
# @property
# def shear_eigval_out(self):
#
#     if self._shear_eigval_out is None:
#         self._shear_eigval_out = self.generate_eigenvalues_type(subtract_density=False)
#
#     return self._shear_eigval_out
#
# @property
# def density_subtracted_eigval_in(self):
#
#     if self._density_subtracted_eigval_in is None:
#         self._density_subtracted_eigval_in = self.generate_eigenvalues_type(subtract_density=True)
#
#     return self._density_subtracted_eigval_in
#
# @property
# def density_subtracted_eigval_out(self):
#
#     if self._density_subtracted_eigval_out is None:
#             self._density_subtracted_eigval_out = self.generate_eigenvalues_type(subtract_density=True)
#     return self._density_subtracted_eigval_out

# def reality_test(shear_class, shear_Fourier, rtol=1e-07):
#     for i in range(int(shear_class.shape)):
#         for j in range(int(shear_class.shape)):
#             for k in range(int(shear_class.shape)):
#                 np.testing.assert_allclose(np.conj(shear_Fourier[i, j, k]), shear_Fourier[-i, -j, -k],
#                                            rtol=rtol,
#                                            err_msg="Error at grid coordinates (i=" + str(i) + ", j="
#                                                    + str(j) + ", k=" + str(k) + ")", verbose=True)
#
#
# def trace_test(shear, density, shape):
#     """ This test takes half an hour """
#     for i in range(shape):
#         for j in range(shape):
#             for k in range(shape):
#                 np.testing.assert_allclose(np.trace(shear[i, j, k]),
#                                            4 * np.pi * G * density[i, j, k],
#                                            err_msg="Error at grid coordinates (i=" + str(i) + ", j="
#                                                    + str(j) + ", k=" + str(k) + ")", verbose=True)
#                 print("Done i = " + str(i) + ", j = " + str(j) + ", k = " + str(k))
#
#
# def trace_test_1d(shear, density, rtol=1e-07):
#     mean = np.mean(density)
#     np.testing.assert_allclose(np.trace(shear, axis1=1, axis2=2),
#                                4 * np.pi * G * (density - mean), rtol=rtol,
#                                # err_msg="Error at grid coordinates (i=" + str(i) + ", j="
#                                #         + str(j) + ", k=" + str(k) + ")",
#                                verbose=True)




        # def shear_fix_nyquist(self, shear_fourier):
        #     shear_fourier_fixed = np.copy(shear_fourier)
        #
        #     sh = int(self.shape)
        #     hsh = int(self.shape/2)
        #
        #     for j in range(1, hsh):
        #
        #         shear_fourier_fixed[0, sh - j, hsh] = np.conj(shear_fourier[0, j, hsh])
        #         shear_fourier_fixed[0, hsh, sh - j] = np.conj(shear_fourier[0, hsh, j])
        #
        #         shear_fourier_fixed[sh - j, hsh, 0] = np.conj(shear_fourier[j, hsh, 0])
        #         shear_fourier_fixed[hsh, sh - j, 0] = np.conj(shear_fourier[hsh, j, 0])
        #
        #         shear_fourier_fixed[sh - j, 0, hsh] = np.conj(shear_fourier[j, 0, hsh])
        #         shear_fourier_fixed[hsh, 0, sh - j] = np.conj(shear_fourier[hsh, 0, j])
        #
        #         for i in range(1, hsh):
        #
        #             shear_fourier_fixed[sh - i, sh - j, hsh] = np.conj(shear_fourier[i, j, hsh])
        #             shear_fourier_fixed[i, sh - j, hsh] = np.conj(shear_fourier[sh - i, j, hsh])
        #             shear_fourier_fixed[i, hsh, hsh] = np.conj(shear_fourier[sh - i, hsh, hsh])
        #
        #             shear_fourier_fixed[sh - i, hsh, sh - j] = np.conj(shear_fourier[i, hsh, j])
        #             shear_fourier_fixed[sh - i, hsh, j] = np.conj(shear_fourier[i, hsh, sh - j])
        #             shear_fourier_fixed[hsh, hsh, i] = np.conj(shear_fourier[hsh, hsh, sh - i])
        #
        #             shear_fourier_fixed[hsh, sh - i, sh - j] = np.conj(shear_fourier[hsh, i, j])
        #             shear_fourier_fixed[hsh, i, sh - j] = np.conj(shear_fourier[hsh, sh - i, j])
        #             shear_fourier_fixed[hsh, i, hsh] = np.conj(shear_fourier[hsh, sh - i, hsh])
        #
        #     return shear_fourier_fixed
    # for i in range(shape):
    #     imirror = i - shape/2
    #     for j in range(np.sqrt((shape/2)**2 - i**2)):
    #         jmirror = j - shape/2
    #         for k in range(np.sqrt((shape/2)**2 - i**2 - j**2)):
    #             kmirror = k - shape/2
    #             if i == 0 and j == 0 and k == 0:
    #                 shear_fourier[i, j, k] = np.zeros((3, 3), dtype=complex)
    #             else:
    #                 shear_fourier[i, j, k] = self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
    #                 shear_fourier[]

    # shear_fourier = np.array([np.zeros((3, 3)) if i == 0 and j == 0 and k == 0 else
    #                           self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
    #                           for k in range(shape) for j in range(shape) for i in range(shape)])
    #
    # shear_fourier = np.array(shear_fourier).reshape((shape, shape, shape, 3, 3))

    # @staticmethod
    # def shear_prefactor_ki_kj(i, j, k, k_mode, boxsize):
    #     k_mode_coordinates = np.zeros((3, 3), dtype=complex)
    #     im = cmath.sqrt(-1)
    #
    #     delta_x = boxsize/ 256
    #     if i == 128:
    #         ki_mode = (np.exp(-im * k_mode[i] * delta_x) - 1)/delta_x
    #     else:
    #         ki_mode = -im * k_mode[i]
    #
    #     if j == 128:
    #         kj_mode = (np.exp(-im * k_mode[j] * delta_x) - 1) / delta_x
    #     else:
    #         kj_mode = -im * k_mode[j]
    #
    #     if k == 128:
    #         kk_mode = (np.exp(-im * k_mode[k] * delta_x) - 1) / delta_x
    #     else:
    #         kk_mode = -im * k_mode[k]
    #
    #     k_mode_coordinates[0] = [ki_mode * ki_mode, ki_mode * kj_mode, ki_mode * kk_mode]
    #     k_mode_coordinates[1] = [kj_mode * ki_mode, kj_mode * kj_mode, kj_mode * kk_mode]
    #     k_mode_coordinates[2] = [kk_mode * ki_mode, kk_mode * kj_mode, kk_mode * kk_mode]
    #
    #     return k_mode_coordinates


    # PREFACTOR CHANGED WITH L = 256
    #
    # @staticmethod
    # def shear_prefactor_ki_kj(i, j, k, k_mode):
    #     k_mode_coordinates = np.zeros((3, 3), dtype=complex)
    #     im = cmath.sqrt(-1)
    #     if i == 128:
    #         ki_mode = (np.exp(-im * k_mode[i]*256) - 1)/256
    #     else:
    #         ki_mode = -im * k_mode[i]
    #
    #     if j == 128:
    #         kj_mode = (np.exp(-im * k_mode[j] * 256) - 1) / 256
    #     else:
    #         kj_mode = -im * k_mode[j]
    #
    #     if k == 128:
    #         kk_mode = (np.exp(-im * k_mode[k] * 256) - 1) / 256
    #     else:
    #         kk_mode = -im * k_mode[k]
    #
    #     k_mode_coordinates[0] = [ki_mode * ki_mode, ki_mode * kj_mode, ki_mode * kk_mode]
    #     k_mode_coordinates[1] = [kj_mode * ki_mode, kj_mode * kj_mode, kj_mode * kk_mode]
    #     k_mode_coordinates[2] = [kk_mode * ki_mode, kk_mode * kj_mode, kk_mode * kk_mode]
    #
    #     return k_mode_coordinates

    # def get_shear_from_potential_in_fourier_space(self, potential, boxsize, shape):
    #     """
    #     Computes the 3x3 shear tensor at each grid point from the gravitational potential.
    #     This function takes ~5.25 minutes to run on a single processor.
    #
    #     """
    #     shape = int(shape)
    #     a_space = np.concatenate((np.arange(shape/2), np.arange(-shape/2, 0)))
    #     k_mode = 2 * np.pi * a_space / boxsize
    #
    #     shear_fourier = np.zeros((shape, shape, shape, 3, 3), dtype=complex)
    #
    #     for i in range(int(shape/2)):
    #         imirror = -i
    #
    #         for j in range(int(shape/2)):
    #             jmirror = -j
    #
    #             for k in range(int(shape/2)):
    #                 kmirror = -k
    #
    #                 if i == 0 and j == 0 and k == 0:
    #                     shear_fourier[i, j, k] = np.zeros((3, 3), dtype=complex)
    #                 else:
    #
    #                     shear_fourier[i, j, k] = self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
    #                     shear_fourier[i, jmirror, k] = self.calculate_shear_in_single_grid(potential, i, jmirror, k,
    #                                                                                        k_mode)
    #                     shear_fourier[i, j, kmirror] = self.calculate_shear_in_single_grid(potential, i, j, kmirror,
    #                                                                                        k_mode)
    #                     shear_fourier[imirror, j, kmirror] = self.calculate_shear_in_single_grid(potential, imirror, j,
    #                                                                                              kmirror, k_mode)
    #                     shear_fourier[imirror, jmirror, k] = self.calculate_shear_in_single_grid(potential, imirror,
    #                                                                                              jmirror, k, k_mode)
    #                     shear_fourier[i, jmirror, kmirror] = self.calculate_shear_in_single_grid(potential, i, jmirror,
    #                                                                                              kmirror,  k_mode)
    #                     shear_fourier[imirror, jmirror, kmirror] = self.calculate_shear_in_single_grid(potential,
    #                                                                                                    imirror,
    #                                                                                                    jmirror,
    #                                                                                                    kmirror, k_mode)
    #     return shear_fourier

    # def get_shear_from_potential_in_fourier_space(self, potential, boxsize, shape):
    #     """
    #     Computes the 3x3 shear tensor at each grid point from the gravitational potential.
    #     This function takes ~5.25 minutes to run on a single processor.
    #
    #     """
    #     shape = int(shape)
    #     a_space = np.concatenate((np.arange(shape/2), np.arange(-shape/2, 0)))
    #     k_mode = 2 * np.pi * a_space / boxsize
    #
    #     shear_fourier = np.zeros((shape, shape, shape, 3, 3), dtype=complex)
    #
    #     for i in range(int(shape/2)):
    #         imirror = -i
    #
    #         for j in range(int(np.floor(np.sqrt((shape/2)**2 - i**2)))):
    #             jmirror = -j
    #
    #             for k in range(int(np.floor(np.sqrt((shape/2)**2 - i**2 - j**2)))):
    #                 kmirror = -k
    #
    #                 if i == 0 and j == 0 and k == 0:
    #                     shear_fourier[i, j, k] = np.zeros((3, 3), dtype=complex)
    #                 else:
    #
    #                     shear_fourier[i, j, k] = self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
    #                     shear_fourier[i, jmirror, k] = self.calculate_shear_in_single_grid(potential, i, jmirror, k,
    #                                                                                        k_mode)
    #                     shear_fourier[i, j, kmirror] = self.calculate_shear_in_single_grid(potential, i, j, kmirror,
    #                                                                                        k_mode)
    #                     shear_fourier[imirror, j, kmirror] = self.calculate_shear_in_single_grid(potential, imirror, j,
    #                                                                                              kmirror, k_mode)
    #                     shear_fourier[imirror, jmirror, k] = self.calculate_shear_in_single_grid(potential, imirror,
    #                                                                                              jmirror, k, k_mode)
    #                     shear_fourier[i, jmirror, kmirror] = self.calculate_shear_in_single_grid(potential, i, jmirror,
    #                                                                                              kmirror,  k_mode)
    #                     shear_fourier[imirror, jmirror, kmirror] = self.calculate_shear_in_single_grid(potential,
    #                                                                                                    imirror,
    #                                                                                                    jmirror,
    #                                                                                                    kmirror, k_mode)
        #     return shear_fourier

    # def get_norm_k_box(shape, boxsize):
    #     a_space = np.concatenate((np.arange(shape/2), np.arange(-shape/2, 0)))
    #     k_mode = 2 * np.pi * a_space / boxsize
    #
    #     im = cmath.sqrt(-1)
    #     delta_x = boxsize / 256
    #
    #     k_norm_squared = np.zeros((shape, shape, shape), dtype=complex)
    #
    #     for i in range(shape):
    #         for j in range(shape):
    #             for k in range(shape):
    #
    #                 if i == 128:
    #                     ki_mode = (np.exp(-im * k_mode[i] * delta_x) - 1)/delta_x
    #                 else:
    #                     ki_mode = -im * k_mode[i]
    #
    #                 if j == 128:
    #                     kj_mode = (np.exp(-im * k_mode[j] * delta_x) - 1) / delta_x
    #                 else:
    #                     kj_mode = -im * k_mode[j]
    #
    #                 if k == 128:
    #                     kk_mode = (np.exp(-im * k_mode[k] * delta_x) - 1) / delta_x
    #                 else:
    #                     kk_mode = -im * k_mode[k]
    #
    #                 k_norm_squared[i,j,k] = abs(ki_mode)**2 + abs(kj_mode)**2 + abs(kk_mode)**2
    #
    #     return k_norm_squared

# @staticmethod
# def shear_prefactor_ki_kj(i, j, k, k_mode):
#     k_mode_coordinates = np.zeros((3, 3))
#
#     if i == 128:
#         ki_mode = - k_mode[-128]
#     else:
#         ki_mode = k_mode[i]
#
#     if j == 128:
#         kj_mode = - k_mode[-128]
#     else:
#         kj_mode = k_mode[j]
#
#     if k == 128:
#         kk_mode = - k_mode[-128]
#     else:
#         kk_mode = k_mode[k]
#
#     k_mode_coordinates[0] = [ki_mode * ki_mode, ki_mode * kj_mode, ki_mode * kk_mode]
#     k_mode_coordinates[1] = [kj_mode * ki_mode, kj_mode * kj_mode, kj_mode * kk_mode]
#     k_mode_coordinates[2] = [kk_mode * ki_mode, kk_mode * kj_mode, kk_mode * kk_mode]
#
#     return k_mode_coordinates

        # def shear_prefactor_ki_kj(self, i, j, k, k_mode, boxsize=None):
        #     k_mode_coordinates = np.zeros((3, 3), dtype=complex)
        #     im = cmath.sqrt(-1)
        #
        #     if boxsize is None:
        #         boxsize = self.boxsize
        #     delta_x = boxsize/ 256
        #
        #     if i == 128:
        #         ki_mode = (np.exp(-im * k_mode[i] * delta_x) - 1)/delta_x
        #     else:
        #         ki_mode = -im * k_mode[i]
        #
        #     if j == 128:
        #         kj_mode = (np.exp(-im * k_mode[j] * delta_x) - 1) / delta_x
        #     else:
        #         kj_mode = -im * k_mode[j]
        #
        #     if k == 128:
        #         kk_mode = (np.exp(-im * k_mode[k] * delta_x) - 1) / delta_x
        #     else:
        #         kk_mode = -im * k_mode[k]
        #
        #     k_mode_coordinates[0] = [ki_mode * ki_mode, ki_mode * kj_mode, ki_mode * kk_mode]
        #     k_mode_coordinates[1] = [kj_mode * ki_mode, kj_mode * kj_mode, kj_mode * kk_mode]
        #     k_mode_coordinates[2] = [kk_mode * ki_mode, kk_mode * kj_mode, kk_mode * kk_mode]
        #
        #     return k_mode_coordinates

    # def get_shear_from_potential_in_fourier_space(self, potential, shape, boxsize):
    #     """
    #     Computes the 3x3 shear tensor at each grid point from the gravitational potential.
    #     This function takes ~5.25 minutes to run on a single processor.
    #
    #     """
    #     shape = int(shape)
    #     a_space = np.concatenate((np.arange(shape/2), np.arange(-shape/2, 0)))
    #     k_mode = 2 * np.pi * a_space / boxsize
    #
    #     shear_fourier = np.zeros((shape, shape, shape, 3, 3), dtype=complex)
    #
    #     for i in range(int((shape/2)+1)):
    #         imirror = -i
    #         for j in range(int((shape/2)+1)):
    #             jmirror = -j
    #             for k in range(int((shape/2)+1)):
    #                 kmirror = -k
    #                 if k == 128:
    #                     kmirror =
    #                 if i == 0 and j == 0 and k == 0:
    #                     shear_fourier[i, j, k] = np.zeros((3, 3), dtype=complex)
    #                 else:
    #                     shear_fourier[i, j, k] = self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
    #                     shear_fourier[-i, -j, -k] = np.conj(shear_fourier[i, j, k])
    #
    #     return shear_fourier

    # @staticmethod
    # def shear_fix_nyquist(shear_fourier, shape):
    #     shear_fourier_fixed = np.copy(shear_fourier)
    #     for i in range(int(shape/2)):
    #         for j in range(int(shape/2)):
    #             shear_fourier_fixed[-i, -j, int(shape/2)] = np.conj(shear_fourier[i, j, int(shape/2)])
    #             shear_fourier_fixed[int(shape/2), -i, -j] = np.conj(shear_fourier[int(shape/2), i, j])
    #             shear_fourier_fixed[-i, int(shape/2),-j] = np.conj(shear_fourier[i, int(shape/2), j])
    #
    #             shear_fourier_fixed[int(shape/2), int(shape/2), -i] = np.conj(shear_fourier[int(shape/2), int(shape/2), i])
    #             shear_fourier_fixed[-i, int(shape/2), int(shape/2)] = np.conj(shear_fourier[i, int(shape/2), int(shape/2)])
    #             shear_fourier_fixed[int(shape/2):, -i, int(shape/2)] = np.conj(shear_fourier[int(shape/2), i, int(shape/2)])
    #
    #     return shear_fourier_fixed

                # @staticmethod
                # def shear_fix_nyquist(shear_fourier, shape):
                #     shear_fourier_fixed = np.copy(shear_fourier)
                #     for i in range(int((shape / 2)+1)):
                #         for j in range(int((shape / 2)+1)):
                #
                #             shear_fourier_fixed[i, -127:, 128] = np.conj(shear_fourier[i, 1:128, 128])[::-1]
                #             shear_fourier_fixed[-127:, 128, 128] = np.conj(shear_fourier[i, 1:128, 128])[::-1]
                #             shear_fourier_fixed[i, -127:, 128] = np.conj(shear_fourier[i, 1:128, 128])[::-1]
                #
                #
                #
                #             shear_fourier_fixed[-i, -j, int(shape / 2)] = np.conj(shear_fourier[i, j, int(shape / 2)])
                #             shear_fourier_fixed[int(shape / 2), -i, -j] = np.conj(shear_fourier[int(shape / 2), i, j])
                #             shear_fourier_fixed[-i, int(shape / 2), -j] = np.conj(shear_fourier[i, int(shape / 2), j])
                #
                #             shear_fourier_fixed[i, -j, int(shape / 2)] = np.conj(shear_fourier[-i, j, int(shape / 2)])
                #             shear_fourier_fixed[int(shape / 2), i, -j] = np.conj(shear_fourier[int(shape / 2), -i, j])
                #             shear_fourier_fixed[i, int(shape / 2), -j] = np.conj(shear_fourier[-i, int(shape / 2), j])
                #
                #             shear_fourier_fixed[i, -j, int(shape / 2)] = np.conj(shear_fourier[i, -j, int(shape / 2)])
                #             shear_fourier_fixed[int(shape / 2), i, -j] = np.conj(shear_fourier[int(shape / 2), i, -j])
                #             shear_fourier_fixed[i, int(shape / 2), -j] = np.conj(shear_fourier[i, int(shape / 2), -j])
                #
                #     return shear_fourier_fixed

    # # TEST IF DIFFERENT METHOD GIVES SAME SHEAR TENSOR -- WORKS
    #
    # def null_test_shear(self, density, shape, boxsize):
    #     t0 = time.clock()
    #     t00 = time.time()
    #
    #     potential = self.get_potential_from_density_fourier_space(density, shape, boxsize)
    #     print("Done potential")
    #     shear_fourier = self.get_shear_from_potential_in_fourier_space(potential, shape, boxsize, True)
    #
    #     s_00 = np.fft.ifftn(shear_fourier[:, :, :, 0, 0])
    #     s_01 = np.fft.ifftn(shear_fourier[:, :, :, 0, 1])
    #     s_02 = np.fft.ifftn(shear_fourier[:, :, :, 0, 2])
    #     s_10 = np.fft.ifftn(shear_fourier[:, :, :, 1, 0])
    #     s_11 = np.fft.ifftn(shear_fourier[:, :, :, 1, 1])
    #     s_12 = np.fft.ifftn(shear_fourier[:, :, :, 1, 2])
    #     s_20 = np.fft.ifftn(shear_fourier[:, :, :, 2, 0])
    #     s_21 = np.fft.ifftn(shear_fourier[:, :, :, 2, 1])
    #     s_22 = np.fft.ifftn(shear_fourier[:, :, :, 2, 2])
    #
    #     shear_real = np.zeros((shape, shape, shape, 3, 3), dtype=complex)
    #     shear_real[:, :, :, 0, 0] = s_00
    #     shear_real[:, :, :, 0, 1] = s_01
    #     shear_real[:, :, :, 0, 2] = s_02
    #     shear_real[:, :, :, 1, 0] = s_10
    #     shear_real[:, :, :, 1, 1] = s_11
    #     shear_real[:, :, :, 1, 2] = s_12
    #     shear_real[:, :, :, 2, 0] = s_20
    #     shear_real[:, :, :, 2, 1] = s_21
    #     shear_real[:, :, :, 2, 2] = s_22
    #
    #     print("Wall time" + str(time.time() - t00))
    #     print("Process time" + str(time.clock() - t0))
    #     return shear_real

