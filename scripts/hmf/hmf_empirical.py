import numpy as np
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
from scripts.hmf import hmf
import pynbody


def get_predicted_mass(log_masses_bins, kernel="PS", initial_parameters=None, trajectories=None):
    masses = 10 ** log_masses_bins
    masses.units = log_masses_bins.units

    if kernel == "ST":
        predicted_mass = mp.get_predicted_analytic_mass(masses, initial_parameters, barrier="ST", cosmology="WMAP5",
                                                        a=0.707, trajectories=trajectories)
    elif kernel == "PS":
        predicted_mass = mp.get_predicted_analytic_mass(masses, initial_parameters, barrier="spherical", trajectories=trajectories)

    else:
        raise AttributeError("Selected wrong kernel")
    return predicted_mass


def get_empirical_total_mass_per_bin(initial_parameters, predicted_mass, log_M_min=10, log_M_max=15, delta_log_M=0.1,
                                     log_m_bins=None):

    if log_m_bins is None:
        log_m_bins = np.arange(log_M_min, log_M_max, delta_log_M)

    num_particles_predicted, bins = np.histogram(np.log10(predicted_mass), bins=log_m_bins)

    m_particle = hmf.get_particle_mass_correct_units(initial_parameters, with_h=True)
    total_mass = num_particles_predicted * m_particle

    m_empirical = 10**((bins[1:] + bins[:-1])/2)
    return m_empirical, total_mass


def total_mass_from_predicted_mass(predicted_mass, bins, initial_parameters=None, with_h=True):
    n_particles, bins = np.histogram(np.log10(predicted_mass), bins=bins)

    particle_mass = hmf.get_particle_mass_correct_units(initial_parameters, with_h=with_h)
    total_mass = n_particles * particle_mass
    return total_mass


def get_predicted_total_mass_in_mass_bins(log_masses_bins, kernel="ST", trajectories=None, initial_parameters=None,
                                          predicted_masses=None):
    if predicted_masses is None:
        predicted_mass = get_predicted_mass(log_masses_bins, kernel, initial_parameters, trajectories=trajectories)

    total_mass = total_mass_from_predicted_mass(predicted_mass, log_masses_bins, initial_parameters)
    return total_mass


def get_predicted_number_of_halos_in_mass_bins(log_masses_bins, log_M_mid, kernel="ST", trajectories=None,
                                               initial_parameters=None, predicted_masses=None):
    if initial_parameters is None:
        initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

    total_mass = get_predicted_total_mass_in_mass_bins(log_masses_bins, kernel=kernel,
                                                       initial_parameters=initial_parameters,
                                                       trajectories=trajectories, predicted_masses=predicted_masses)
    number_halos = hmf.get_number_halos_from_bins_total_mass(total_mass, log_M_mid)

    return number_halos


def get_predicted_number_density_halos_in_mass_bins(log_masses_bins, log_M_mid, kernel="ST", initial_parameters=None,
                                                    trajectories=None, predicted_masses=None):
    number_halos = get_predicted_number_of_halos_in_mass_bins(log_masses_bins, log_M_mid, kernel=kernel,
                                                              trajectories=trajectories,
                                                              initial_parameters=initial_parameters,
                                                              predicted_masses=predicted_masses)

    number_density = hmf.get_comoving_number_density_halos(number_halos, initial_parameters)
    return number_density


def get_number_halos_from_number_particles_binned(num_particles_predicted, bins, initial_parameters):

    m_particle = hmf.get_particle_mass_correct_units(initial_parameters, with_h=True)
    total_mass = num_particles_predicted * m_particle

    mid_log = ((bins[1:] + bins[:-1]) / 2).view(pynbody.array.SimArray)
    mid_log.units = "Msol h^-1"

    number_halos = hmf.get_number_halos_from_bins_total_mass(total_mass, mid_log)
    return 10**mid_log, number_halos


def get_empirical_number_halos(predicted_mass, initial_parameters, log_M_min=10, log_M_max=15, delta_log_M=0.1,
                               log_m_bins=None):
    if log_m_bins is None:
        log_m_bins = np.arange(log_M_min, log_M_max, delta_log_M)

    num_particles_predicted, bins = np.histogram(np.log10(predicted_mass), bins=log_m_bins)
    m, number_halos = get_number_halos_from_number_particles_binned(num_particles_predicted, bins, initial_parameters)
    return number_halos


def get_empirical_total_mass(initial_parameters, predicted_mass, log_M_min=10, log_M_max=15, delta_log_M=0.1,
                             log_m_bins=None):
    if log_m_bins is None:
        log_m_bins = np.arange(log_M_min, log_M_max, delta_log_M)

    num_particles_predicted, bins = np.histogram(np.log10(predicted_mass), bins=log_m_bins)

    m_particle = hmf.get_particle_mass_correct_units(initial_parameters, with_h=True)
    total_mass = num_particles_predicted * m_particle

    mid_log = ((log_m_bins[1:] + log_m_bins[:-1]) / 2).view(pynbody.array.SimArray)
    mid_log.units = "Msol h^-1"

    number_halos = hmf.get_number_halos_from_bins_total_mass(total_mass, mid_log)
    n_den = hmf.get_comoving_number_density_halos(number_halos)
    dndlog10 = hmf.get_dn_dlog10m(n_den, 10 ** log_m_bins, 10 ** mid_log)
    return 10**mid_log, dndlog10
