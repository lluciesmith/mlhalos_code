import sys

sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
import scipy
from scipy.misc import factorial
from mlhalos import distinct_colours
import scipy.integrate
import scipy.stats
import scipy.interpolate
from scripts.hmf import hmf_simulation as hmf_sim
from scripts.hmf import hmf_theory
from scripts.hmf.larger_sim import hmf_analysis as ha
from mlhalos import parameters


############################ LIKELIHOOD AND CHI - SQUARED ############################


def poisson_distribution(x, nu):
    p = nu**x * np.exp(-nu) / factorial(x)
    return p


def log_poisson_likelihood(x, nu):
    """ This is -2 * log(likelihood) for a Poisson likelihood """
    L = x * np.log(nu) - nu - np.log(factorial(x))
    return -2*L


def log_gaussian_likelihood(x, nu, sigma):
    """ This is -2 * log(likelihood) for a Gaussian likelihood """
    L = np.log(2 * np.pi * sigma**2) + ((x - nu) / sigma)**2
    return L


def log_likelihood(data, mean):
    m, var, sk, kurtosis = scipy.stats.poisson.stats(mean, moments="mvsk")
    r1 = sk/m
    r2 = kurtosis/m
    if mean != 0:

        if (r1 < 1e-2) and (r2 < 1e-2):
            l = log_gaussian_likelihood(data, mean, np.sqrt(var))
            # print("Gaussian " + str(mean))
        else:
            l = log_poisson_likelihood(data, mean)
            # print("Poisson " + str(mean))
    else:
        l = 0
    return l


def chi_squared_each_bin(data, mean):
    log_l = np.array([log_likelihood(data[i], mean[i]) for i in range(len(mean))])
    norm = np.array([np.log(2 * np.pi* mean[i]) if mean[i]!=0 else 0 for i in range(len(mean))])
    chisquar_bins = log_l - norm
    return chisquar_bins


def chi_squared(data, mean):
    chisquar_bins = chi_squared_each_bin(data, mean)
    return np.sum(chisquar_bins)


def chi_squared_gaussian(data_points, mean, sigma):
    log_l = np.array([log_gaussian_likelihood(data_points, mean[i], sigma)  for i in range(len(mean))])
    norm_g = np.log(2 * np.pi* mean)
    chisquar_gaus = np.sum(log_l) - norm_g
    return chisquar_gaus


############################ HMF LIKELIHOOD ############################


def get_empirical_number_density_halos(kernel, boxsize, initial_parameters=None):
    if boxsize == 50:
        if initial_parameters is None:
            initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

        if kernel == "PS":
            predicted_mass = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k"
                                           "/ALL_predicted_masses_1500_even_log_m_spaced.npy")
        elif kernel == "ST":
            predicted_mass = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k"
                                               "/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
        else:
            raise NameError

    elif boxsize == 200:
        if initial_parameters is None:
            initial_snapshot = "/Users/lls/Documents/CODE/larger_sim/sim200.gadget3"
            initial_parameters = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                                           path="/Users/lls/Documents/CODE/")
        if kernel == "PS":
            predicted_mass = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")
        elif kernel == "ST":
            predicted_mass = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_ST_predicted_masses.npy")
        else:
            raise NameError
    else:
        raise NameError

    emp_num_density_halos = ha.get_empirical_number_density_halos(predicted_mass, initial_parameters, boxsize)
    return emp_num_density_halos


def get_likelihood(kernel, boxsize, initial_parameters):

    m_theory, num_theory = ha.get_theory_number_density_halos(kernel, initial_parameters, boxsize=boxsize)
    num_theory *= (boxsize**3)

    n_emp = get_empirical_number_density_halos(kernel, boxsize, initial_parameters)
    n_emp *= (boxsize ** 3)

    log_l = np.array([log_likelihood(n_emp[i], num_theory[i]) for i in range(len(num_theory))])
    return np.sum(log_l)


def get_relative_likelihood_small_large_box(kernel, initial_parameters_small, small_boxsize=50, large_boxsize=200,
                                            return_individual=True):
    m_theory, num_theory = ha.get_theory_number_density_halos(kernel, initial_parameters_small, boxsize=small_boxsize)
    num_theory *= (small_boxsize**3)

    n_emp_small = get_empirical_number_density_halos(kernel, small_boxsize, initial_parameters_small)
    n_emp_small *= (small_boxsize ** 3)

    n_emp_large = get_empirical_number_density_halos(kernel, large_boxsize)
    n_emp_large *= (small_boxsize ** 3)

    log_l_small = np.array([log_likelihood(n_emp_small[i], num_theory[i]) for i in range(len(num_theory))])
    log_l_large = np.array([log_likelihood(n_emp_large[i], num_theory[i]) for i in range(len(num_theory))])

    if return_individual is True:
        return np.sum(log_l_small), np.sum(log_l_large), np.sum(log_l_small)/np.sum(log_l_large)
    else:
        return np.sum(log_l_small)/np.sum(log_l_large)


def get_relative_chi_squared_small_large_box(kernel, initial_parameters_small, small_boxsize=50, large_boxsize=200,
                                            return_individual=True):
    m_theory, num_theory = ha.get_theory_number_density_halos(kernel, initial_parameters_small, boxsize=small_boxsize)
    num_theory *= (small_boxsize**3)

    n_emp_small = get_empirical_number_density_halos(kernel, small_boxsize, initial_parameters_small)
    n_emp_small *= (small_boxsize ** 3)

    n_emp_large = get_empirical_number_density_halos(kernel, large_boxsize)
    n_emp_large *= (small_boxsize ** 3)

    m_sim, num_sim = hmf_sim.get_true_number_halos_per_mass_bins(initial_parameters_small, np.arange(10, 15, 0.1))
    restr = (num_sim >= 10) & (m_sim >= 8.23 * 10 ** 10)

    csq_small = chi_squared(n_emp_small[restr], num_theory[restr])
    csq_large = chi_squared(n_emp_large[restr], num_theory[restr])
    return csq_small, csq_large


def get_distribution_number_halos_in_mass_bin(function, lower_bin_limit, higher_bin_limit):
    #m_finer = np.linspace(lower_bin_limit, higher_bin_limit, 1000)
    # m_finer_mid = (m_finer[1:] + m_finer[:-1]) / 2
    m_finer = np.linspace(np.log10(lower_bin_limit), np.log10(higher_bin_limit), 1000)

    delta_m = np.diff([lower_bin_limit, higher_bin_limit])

    # num_density_halos = function(m_finer_mid) * delta_m
    num_density_halos = 10**function(m_finer) * delta_m
    return num_density_halos


def interpolate_dndm(initial_parameters,  kernel="PS", log_M_min=8, log_M_max=16, delta_log_M=0.1):
    m_larger, dndm_PS_larger = hmf_theory.get_dndm_theory(initial_parameters, kernel=kernel, log_M_min=log_M_min,
                                                                      log_M_max=log_M_max, delta_log_M=delta_log_M)
    function = scipy.interpolate.interp1d(m_larger, dndm_PS_larger)
    return function


def interpolate_log_dndm(initial_parameters,  kernel="PS", log_M_min=8, log_M_max=16, delta_log_M=0.1):
    m_larger, dndm_PS_larger = hmf_theory.get_dndm_theory(initial_parameters, kernel=kernel, log_M_min=log_M_min,
                                                                      log_M_max=log_M_max, delta_log_M=delta_log_M)
    function = scipy.interpolate.interp1d(np.log10(m_larger), np.log10(dndm_PS_larger))
    return function


def get_theory_distribution_per_mass_bin(initial_parameters, kernel="PS", log_M_min=10, log_M_max=15, delta_log_M=0.1,
                                         boxsize=50.):
    bins = 10 ** np.arange(log_M_min, log_M_max, delta_log_M)
    volume = initial_parameters.boxsize_comoving**3

    # f_int = interpolate_dndm(initial_parameters,  kernel=kernel, log_M_min=log_M_min-1, log_M_max=log_M_max+1,
    #                          delta_log_M=delta_log_M)
    f_int_log = interpolate_log_dndm(initial_parameters,  kernel=kernel, log_M_min=log_M_min-1,
                                     log_M_max=log_M_max+1, delta_log_M=delta_log_M)

    num_density = np.array([get_distribution_number_halos_in_mass_bin(f_int_log, bins[i], bins[i+1])
                            for i in range(len(bins) - 1)])
    n = num_density * volume
    return n


if __name__ == "__main__":
    ic_50 = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    ic_200 = parameters.InitialConditionsParameters(
        initial_snapshot="/Users/lls/Documents/CODE/larger_sim/sim200.gadget3",  path="/Users/lls/Documents/CODE/")

    small_PS, large_PS, rel_PS = get_relative_likelihood_small_large_box("PS", ic_50, small_boxsize=50,
                                                                         large_boxsize=200, return_individual=True)

    small_ST, large_ST, rel_ST = get_relative_likelihood_small_large_box("ST", ic_50, small_boxsize=50,
                                                                         large_boxsize=200, return_individual=True)

    np.save("/Users/lls/Documents/CODE/stored_files/hmf/likelihood/small_PS_likelihood.npy", small_PS)
    np.save("/Users/lls/Documents/CODE/stored_files/hmf/likelihood/large_PS_likelihood.npy", large_PS)
    np.save("/Users/lls/Documents/CODE/stored_files/hmf/likelihood/small_ST_likelihood.npy", small_ST)
    np.save("/Users/lls/Documents/CODE/stored_files/hmf/likelihood/large_ST_likelihood.npy", large_ST)
