import numpy as np
from mlhalos import shear
from mlhalos import density


def density_test():
    shear_scale = range(50)
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=shear_scale)
    d = density.Density(path="/Users/lls/Documents/CODE/", shear_scale=shear_scale)

    # Density

    density_fourier = s._density_scale_fourier
    den_shear = np.array([np.real(np.fft.ifftn(density_fourier[i]).reshape(s.shape**3,))
                          for i in shear_scale])
    den_correct = d._density

    np.testing.assert_allclose(den_correct.transpose(), den_shear)
    print("Passed density test")

    # Density contrast

    contrast_shear_fourier = np.array([s.rescale_density_to_density_contrast_in_fourier_space(density_fourier[i])
                                       for i in shear_scale])
    contrast_shear = np.array([np.real(np.fft.ifftn(contrast_shear_fourier[i]).reshape(s.shape**3,))
                               for i in shear_scale])
    mean_density = d._mean_density
    contrast_correct = (den_correct - mean_density)/mean_density

    np.testing.assert_allclose(contrast_correct, contrast_shear)
    print("Passed density contrast test")


def trace_test_fourier_space():
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=20)
    den = s._density_scale_fourier
    den_contrast = s.rescale_density_to_density_contrast_in_fourier_space(den).reshape(s.shape**3, )

    phi = s.get_potential_from_density_fourier_space(den, s.shape, s.boxsize)
    shear_fourier = s.get_shear_from_potential_in_fourier_space(phi, s.shape, s.boxsize).reshape(s.shape**3, 3, 3)

    np.testing.assert_allclose(np.trace(shear_fourier, axis1=1, axis2=2), den_contrast, verbose=True, rtol=1e-07)


def trace_test_real_space():
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=20)
    den = s._density_scale_fourier
    den_contrast = s.rescale_density_to_density_contrast_in_fourier_space(den)
    den_contrast_real = np.real(np.fft.ifftn(den_contrast).reshape(s.shape**3, ))

    shear_real = s.get_shear_tensor_at_scale(den)
    np.testing.assert_allclose(np.trace(shear_real, axis1=1, axis2=2), den_contrast_real, verbose=True, rtol=1e-07)


def reality_test_fourier_space():
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=20)
    den = s._density_scale_fourier
    phi = s.get_potential_from_density_fourier_space(den, s.shape, s.boxsize)
    shear_fourier = s.get_shear_from_potential_in_fourier_space(phi, s.shape, s.boxsize)

    for i in range(s.shape):
        for j in range(s.shape):
            for k in range(s.shape):
                np.testing.assert_allclose(np.conj(shear_fourier[i, j, k]), shear_fourier[-i, -j, -k])


def reality_test_real_space():
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=range(50))
    densities = s._density_scale_fourier

    for i in range(densities.shape[0]):
        den = densities[i]
        assert den.shape == (s.shape, s.shape, s.shape)

        phi = s.get_potential_from_density_fourier_space(den, s.shape, s.boxsize)
        shear_fourier = s.get_shear_from_potential_in_fourier_space(phi, s.shape, s.boxsize)

        shear_real = np.fft.ifftn(shear_fourier, axes=(0, 1, 2)).reshape((s.shape ** 3, 3, 3))
        assert np.imag(shear_real).max() < 1e-15


def test_sum_eigenvalues():
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=20)
    den = s._density_scale_fourier
    den_contrast = s.rescale_density_to_density_contrast_in_fourier_space(den)
    den_contrast_real = np.real(np.fft.ifftn(den_contrast).reshape(s.shape**3, ))

    eig = s.shear_eigenvalues

    np.testing.assert_allclose(np.sum(eig, axis=1), den_contrast_real, verbose=True, rtol=1e-06)


def test_sum_eigenvalues_for_all_50_ranges():
    s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=range(50))
    eig = s.shear_eigenvalues

    sum_eig = np.zeros((s.shape**3, 50))
    for i in range(50):
        eig_i = eig[:, int(3 * i):int(3 * i) + 3]
        sum_eig[:, i] = np.sum(eig_i, axis=1)

    d = density.Density(path="/Users/lls/Documents/CODE/")
    densities = d._density
    mean_densities = d._mean_density
    contrasts = (densities - mean_densities)/mean_densities

    np.testing.assert_allclose(sum_eig, contrasts, verbose=True, rtol=1e-06)


# def test_sum_eigenvalues_in():
#     s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=20)
#     den = s._density_scale_fourier
#     den_contrast = s.rescale_density_to_density_contrast_in_fourier_space(den)
#     den_contrast_real = np.real(np.fft.ifftn(den_contrast).reshape(s.shape**3, ))
#
#     eig = s.shear_eigenvalues
#
#     np.testing.assert_allclose(np.sum(eig, axis=1), den_contrast_real, verbose=True)
#
#
# def test_sum_eigenvalues_out():
#     s = shear.Shear(path="/Users/lls/Documents/CODE/", shear_scale=20)
#     den = s._density_scale_fourier
#     den_contrast = s.rescale_density_to_density_contrast_in_fourier_space(den)
#     den_contrast_real = np.real(np.fft.ifftn(den_contrast).reshape(s.shape**3, ))
#
#     eig = s.shear_eigenvalues
#
#     np.testing.assert_allclose(np.sum(eig, axis=1), den_contrast_real, verbose=True)


# def trace_test(shear, density, shape, type="real", rtol=1e-05):
#     for i in range(shape):
#         for j in range(shape):
#             for k in range(shape):
#                 if type == "real":
#                     den_mean = np.mean(density)
#                     np.testing.assert_allclose(np.trace(shear, axis1=1, axis2=2),
#                                                4 * np.pi * G * (density - den_mean), verbose=True, rtol=rtol)
#                 else:
#                     density1 = np.copy(density)
#                     density1[0, 0, 0] = 0
#                     a = np.allclose(np.trace(shear[i, j, k]), 4 * np.pi * G * density1[i, j, k])
#                     if a is False:
#                         print("False")

# def shear_test(initial_parameters=parameters.InitialConditionsParameters(), num_filtering_scales=50,
#                  snapshot=None, shear_scale=27, density_particles=None):
#     """
#     This is unit test which assures that the trace of the shear tensor is the density.
#     """
#
#     def calculate_shear_in_single_grid_with_test(shear_class, potential, i, j, k):
#         a_space = np.concatenate((np.arange(shear_class.shape / 2), np.arange(-shear_class.shape / 2, 0)))
#         k_mode = 2 * np.pi * a_space / shear_class.boxsize
#
#         shear_single_grid = shear_class.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
#         return shear_single_grid
#
#
#     s = shear.Shear(initial_parameters, num_filtering_scales,
#                     snapshot, shear_scale, density_particles)
#
#     density = s._density_scale
#     potential = s.get_potential_from_density_fourier_space(density, s.shape, s.boxsize)
#
#     for i in range(s.shape):
#         for j in range(s.shape):
#             for k in range(s.shape):
#                 if i == 0 and j == 0 and k == 0:
#                     a = np.zeros((3, 3))
#                 else:
#                     shear_single_grid = calculate_shear_in_single_grid_with_test(s, potential, i, j, k)
#                     np.testing.assert_allclose(np.trace(shear_single_grid),
#                                                4 * np.pi * G * density[i, j, k],
#                                                err_msg="Error at grid coordinates (i=" + str(i) + ", j="
#                                                        + str(j) + ", k=" + str(k) + ")", verbose=True)
#
# def reality_test(shear_class, shear_matrix,atol=1e-06, rtol=1e-07, type="real"):
#     for i in range(int(shear_class.shape)):
#         for j in range(int(shear_class.shape)):
#             for k in range(int(shear_class.shape)):
#                 if type == "fourier":
#                     a = np.allclose(np.conj(shear_matrix[i, j, k]), shear_matrix[-i, -j, -k])
#                     if a is False:
#                         print("False")
#                 else:
#                     assert np.imag(shear_matrix).max() < 1e-09
