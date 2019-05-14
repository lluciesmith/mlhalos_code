import numpy as np
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import scipy.integrate
import math
from mlhalos import window
from mlhalos import parameters
import scipy.special
import scipy.interpolate
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from scripts.hmf import predict_masses as mp


def test_with_sigma8(initial_parameters):
    R_8 = 8
    V_8 = 4 / 3 * np.pi * R_8**3
    L_8 = scipy.special.cbrt(V_8)
    l_8 = L_8 * 2

    shape = 256
    boxsize = 50

    # k_x = ssc.k_1D(boxsize, shape)
    # #k_vector = ssc.get_all_ks(shape, boxsize)
    # min_k = abs(k_x).min()
    # max_k = abs(k_x).max()

    k_vector = ssc.get_all_ks(shape, boxsize)
    min_k = k_vector.min()
    max_k = k_vector.max()

    pwspectrum = eb.get_power_spectrum("WMAP5", initial_parameters, z=0)
    k_1d = ssc.k_1D(boxsize, shape)
    W = ssc.window_cube_squared(k_1d, k_1d, k_1d, l_8)
    #k_vector = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    #p_k = pwsp(k_vector)

    #W = window_tophat(k_vector, L)

    integ = lambda k: k**2 * pwspectrum(k) * W(k)

    #D_a_squared = pwspectrum._lingrowth

    #f = ssc.integrand
    # result, error = scipy.integrate.quad(integ, min_k, max_k, args=(l_8, pwspectrum))
    result, error = scipy.integrate.quad(integ, min_k, max_k)
    # result, error = scipy.integrate.nquad(f, [[min_k, max_k], [min_k, max_k], [min_k, max_k]],
    #                                       args=(l_8, pwspectrum))
    #r = result / pwspectrum._lingrowth / (2*np.pi)**3 * V_8
    #result = result * (V_8**3)
    r = result /(2 * math.pi**2) / pwspectrum._lingrowth**2

    k_z = k_y = k_x = ssc.k_1D(boxsize, shape)
    W = ssc.window_cube_squared(k_x, k_y, k_z, boxsize)
    integ = lambda a: a ** 2 * pwspectrum(a) * W(a)
    result, error = scipy.integrate.nquad(f, min_k, max_k)

    result, error = scipy.integrate.nquad(f, [[min_k, max_k], [min_k, max_k], [min_k, max_k]],
                                          args=(l_8, pwspectrum))
    sig_8 = np.sqrt(variance)

    #v_8 = variance_background_mode(l, initial_parameters)


    return v_8


def get_sigma8(pwspectrum):
    shape = 256
    boxsize = 50

    max_k = np.pi * shape / boxsize
    min_k = np.pi / boxsize

    variance, error = scipy.integrate.nquad(sigma_8_integrand, [[min_k, max_k], [min_k, max_k], [min_k, max_k]],
                                          args=(8, pwspectrum), limit=500)
    sig_8 = np.sqrt(8 * variance)
    return sig_8


def sigma_8_integrand(k_x, k_y, k_z, L, power_spectrum):
    # if k_x==0 and k_y==0 and k_z==0:
    #     pass
    # else:

    k = np.sqrt(k_x**2 + k_y**2 + k_z**2)

    integ = power_spectrum(k) * abs(window_tophat(k * L))**2
    return integ


def pspec(k_x, k_y, k_z, power_spectrum):
    k = np.sqrt(k_x ** 2 + k_y ** 2 + k_z ** 2)
    return power_spectrum(k)


def integral_3d(length, ic):
    shape = 256
    boxsize = 50

    max_k = np.pi * shape / boxsize
    min_k = np.pi / boxsize
    pwspectrum = eb.get_power_spectrum("WMAP5", ic, z=0)

    integrand = lambda k_1, k_2, k_3: pspec(k_1, k_2, k_3, pwspectrum) * sinc_2(k_1, k_2, k_3, length)
    variance, error = scipy.integrate.nquad(integrand, [[min_k, max_k], [min_k, max_k], [min_k, max_k]],
                                            opts={'limit': 50})
    return variance


def convolution(shape, boxsize, initial_parameters):
    all_k = get_all_ks(shape, boxsize)

    pws = eb.get_power_spectrum("WMAP5", initial_parameters, z=0)
    pk = pws(all_k)
    a = np.convolve(pk, window)
    return a


#
#
# def interpolated_window_cube_squared(k_x, k_y, k_z, L):
#     W = np.zeros((len(k_x), len(k_y), len(k_z)))
#     for x in range(len(k_x)):
#         for y in range(len(k_y)):
#             for z in range(len(k_z)):
#                 W[x, y, z] = sinc_2(k_x[x], k_y[y], k_z[z], L/2)
#
#     k_vector = np.sqrt(k_x ** 2 + k_y ** 2 + k_z ** 2)
#     interp_W = scipy.interpolate.interp1d(np.log(W), np.log(k_vector))
#     return interp_W


def window_tophat(kR):
    top_hat = (3. * (np.sin(kR) - (kR * np.cos(kR)))) / (kR ** 3)
    return top_hat


def get_all_ks(shape, boxsize):
    if shape == 256:
        a = np.load("/Users/lls/Documents/CODE/stored_files/Fourier_transform_matrix.npy")
    elif shape == 512:
        a = np.load("/Users/lls/Documents/CODE/stored_files/Fourier_transform_matrix_shape_512.npy")
    else:
        raise NameError("Wrong shape!")

    k = 2 * np.pi * a / boxsize
    #k[0,0,0] = 1e-05
    return k



######################## Window function functions ########################


def get_k_1D(L, shape):
    a = np.concatenate((np.arange(0, shape/2), np.arange(-shape/2, 0)))
    k = 2*np.pi*a/L
    return k


def half_nyquist_mode_1D(boxsize, shape):
    k_1d = get_k_1D(boxsize, shape)
    k_half_nyquist = abs(k_1d).max() / 2
    return k_half_nyquist


def half_nyquist_mode_3D(boxsize, shape):
    k_all = get_all_ks(shape, boxsize)
    k_half_nyquist = abs(k_all).max() / 2
    return k_half_nyquist


def sinc_2(k_x, k_y, k_z, l):
    return (np.sinc(k_x * l / np.pi)**2) * (np.sinc(k_y * l/ np.pi)**2) * (np.sinc(k_z * l / np.pi)**2)


def get_window_squared_cube(min_k, max_k, boxsize, num_k=500):
    k_finer = np.linspace(min_k, max_k, num_k, endpoint=True)

    W_flat = np.array([sinc_2(i, j, k, boxsize/2) for i in k_finer for j in k_finer for k in k_finer])
    k_vector = np.array([np.sqrt(i ** 2 + j ** 2 + k ** 2) for i in k_finer for j in k_finer for k in k_finer])
    return k_vector, W_flat


def window_squared_finer_evaluation(shape, boxsize, number_of_ks=500):
    k_max = half_nyquist_mode_1D(boxsize, shape)

    k_vector, W_flat = get_window_squared_cube(0, k_max, boxsize, num_k=number_of_ks)
    return k_vector, W_flat


def get_window_function_squared_box_interpolated(shape, boxsize, number_of_ks=500):
    k_vector, W_flat = window_squared_finer_evaluation(shape, boxsize, number_of_ks=number_of_ks)

    k_un, ind = np.unique(k_vector, return_index=True)
    int_W = scipy.interpolate.interp1d(k_un, W_flat[ind])
    return int_W


# def variance_background_mode(boxsize, initial_parameters, pwspectrum=None, z=99, cosmology="WMAP5", shape=256):
#     k_vector = get_all_ks(shape, boxsize)
#     min_k = k_vector.min()
#     max_k = k_vector.max()
#     k_x = k_1D(boxsize, shape)
#
#     if pwspectrum is None:
#         pwspectrum = eb.get_power_spectrum(cosmology, initial_parameters, z=z)
#
#     f = integrand
#     result, error = scipy.integrate.nquad(f, [[min_k, max_k], [min_k, max_k], [min_k, max_k]],
#                                           args=(boxsize, pwspectrum))
#     result = 8 * result /(2 * math.pi**2) ** 3 / pwspectrum._lingrowth
#
#     # divide by linear growth
#     #result /= pwspectrum._lingrowth
#     return result


def get_reliable_k_modes(k_original):
    k_unique = np.unique(k_original)

    # Cut above half the Nyquist
    half_nyquist = k_unique.max() / 2
    k_unique_half_ny = k_unique[k_unique <= half_nyquist]
    return k_unique_half_ny


def window_cube_squared_fourier_space(L, shape):
    k_1d = get_k_1D(L, shape)

    W = np.zeros((len(k_1d), len(k_1d), len(k_1d)))
    for x in range(len(k_1d)):
        for y in range(len(k_1d)):
            for z in range(len(k_1d)):
                W[x, y, z] = (np.sinc(k_1d[x] * L/2)**2) * (np.sinc(k_1d[y] * L / 2)**2) * (np.sinc(k_1d[z] * L /2)**2)
    return W


def get_window_and_k_modes_reliable(k_orig, W_orig):
    flat_k = np.copy(k_orig.flatten())
    #flat_k[0] = 0

    un_k, ind = np.unique(flat_k, return_index=True)
    k_half_nyquist = np.where(un_k <= un_k.max()/2)[0]
    reliable_k = un_k[k_half_nyquist]

    flat_W = np.copy(W_orig.flatten())
    #flat_W[0] = 0
    W_reliable = abs(flat_W[ind][k_half_nyquist])
    return reliable_k, W_reliable


def power_spectrum_modes(k, pws):
    k_pk = k.copy()
    k_pk[0] = 1

    Pk = pws(k_pk)
    Pk[0] = 0
    return Pk


######################## Variance integral calculation ########################


def get_variance(powerspectrum, window_function, k_maximum, log_interpolated_W=True):
    if log_interpolated_W is True:
        integrand = lambda k: k ** 2 * powerspectrum(k) * np.exp(window_function(np.log(k)))
    else:
        integrand = lambda k: k ** 2 * powerspectrum(k) * window_function(k)

    integrand_ln_k = lambda k: np.exp(k) * integrand(np.exp(k))
    v = scipy.integrate.romberg(integrand_ln_k, math.log(powerspectrum.min_k), math.log(k_maximum),
                                divmax=10, rtol=1.e-4) / (2 * math.pi ** 2)
    return v


######################## Global function ########################


def compute_variance_mean_density_of_box(boxsize, shape, initial_parameters=None, pwspectrum=None, z=0):
    if initial_parameters is None:
        initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

    if pwspectrum is None:
        pwspectrum = eb.get_power_spectrum("WMAP5", initial_parameters, z=z)
    window_function = get_window_function_squared_box_interpolated(shape, boxsize, number_of_ks=500)

    k_max = half_nyquist_mode_3D(boxsize, shape)
    var = get_variance(pwspectrum, window_function, k_max, log_interpolated_W=False)
    print("Variance is " + str(var) + " for boxsize of " + str(boxsize))
    return var


def mean_density_contrast_realisations(boxsize, shape, initial_parameters, num_realisations=30):
    variance = compute_variance_mean_density_of_box(boxsize, shape, initial_parameters)
    std = np.sqrt(variance)
    p = np.random.normal(1, std, size=num_realisations)
    return p


def trajectories_with_new_mean(trajectory, rho_mean_deltas):
    trajectory_new = np.zeros((len(rho_mean_deltas), len(trajectory)))

    for i in range(len(rho_mean_deltas)):
        trajectory_new[i, :] = trajectory + rho_mean_deltas[i]

    return trajectory_new


###### Modify spherical collapse barrier

# def ssc_barrier(masses, initial_parameters, barrier=barrier, cosmology=cosmology, a=a):



def get_predicted_analytic_mass_including_ssc_barrier(masses, initial_parameters, barrier="ST", cosmology="WMAP5",
                                                    a=0.707, trajectories=None):
    barrier_value = ssc_barrier(masses, initial_parameters, barrier=barrier, cosmology=cosmology, a=a)
    print(barrier_value)
    mass_first_upcrossing = mp.record_mass_first_upcrossing(barrier_value, masses,
                                                         trajectories=trajectories,
                                                         halo_min=None)
    return mass_first_upcrossing



def predict_mass_super_sampled_trajectory(trajectory, rho_mean_deltas, mass_bins, initial_parameters, barrier="PS"):
    trajectories_new = trajectories_with_new_mean(trajectory, rho_mean_deltas)
    pred_mass = mp.get_predicted_analytic_mass(mass_bins, initial_parameters, barrier=barrier, cosmology="WMAP5",
                                               trajectories=trajectories_new)
    return pred_mass


def add_k0_mode_to_trajectories(trajectories, boxsize, shape, initial_parameters, num_realisations=30):
    rho_means = mean_density_contrast_realisations(boxsize, shape, initial_parameters, num_realisations=num_realisations)
    new_traj_particles = np.zeros((trajectories.shape[0], len(rho_means), trajectories.shape[1]))

    for i in range(len(trajectories)):
        traj = trajectories[i]
        new_traj_particles[i, :, :] = trajectories_with_new_mean(traj, rho_means)

    rho = initial_parameters.initial_conditions['rho']
    #p = mean_density_contrast_realisations(initial_parameters, variance)
    mean = np.zeros(())
    for number in p:
        rho_prime = rho + number
        m = np.mean(rho_prime)

    return new_traj_particles


if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    pwspectrum = eb.get_power_spectrum("WMAP5", ic, z=0)

    boxsizes = np.array([12.89, 25, 50, 200, 250, 500, 1000])
    shape = 256
    v = np.array([compute_variance_mean_density_of_box(length, shape, initial_parameters=ic, pwspectrum=pwspectrum)
                  for length in boxsizes])
    # shape = 256
    #
    # for i in range(len(boxsizes)):
    #     boxsize = boxsizes[i]
    #
    #     k = get_all_ks(shape, boxsize)
    #     W2 = window_cube_squared_fourier_space(boxsize, shape)
    #     k_rel, W2_rel = get_window_and_k_modes_reliable(k, W2)
    #
    #     int_W = scipy.interpolate.interp1d(np.log(k_rel), np.log(W2_rel))
    #     v = get_variance(pwspectrum, int_W, k_rel.max(), log_interpolated_W=True)
    # # #print("Variance " + str(v[i]) + " for boxsize of length L = " + str(boxsize) + " Mpc/h")
    #
    # #boxsizes = np.array([25, 50, 200, 250, 500, 1000])
    # #v = np.zeros(len(boxsizes))
    #
    # #for i in range(len(boxsizes)):
    # shape = 256
    # #boxsize = boxsizes[i]
    # boxsize = 16
    #
    # k = get_all_ks(shape, boxsize)
    # W2 = window_cube_squared_fourier_space(boxsize, shape)
    # k_rel, W2_rel = get_window_and_k_modes_reliable(k, W2)
    # k_rel[0,0,0] = 0
    #
    # # INTERPOLATING IN LOG SCALE GIVES TWO ORDERS OF MAG DIFFERENCE THAN INTERPOLATING LINEARLY. WHY?
    # int_W = scipy.interpolate.interp1d(np.log(k_rel), np.log(W2_rel))
    # v = get_variance(pwspectrum, int_W, k_rel.max(), log_interpolated_W=True)
    # #print("Variance " + str(v[i]) + " for boxsize of length L = " + str(boxsize) + " Mpc/h")


