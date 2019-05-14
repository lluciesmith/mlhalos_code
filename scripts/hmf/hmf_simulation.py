import numpy as np
from scripts.hmf import hmf
import pynbody


############################ SIMULATION'S HMF FROM HALO MASS OF EACH PARTICLE ############################

def load_halo_mass_particles(initial_parameters):
    halo_mass_particles = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy").view(pynbody.array.SimArray)
    h = initial_parameters.final_snapshot.properties['h']

    halo_m = halo_mass_particles * h
    halo_m.units = "Msol h^-1"
    return halo_m


def get_simulation_total_mass(initial_parameters, log_M_min=10, log_M_max=15, delta_log_M=0.1, log_m_bins=None):
    halo_mass_particles = load_halo_mass_particles(initial_parameters)
    h_mass_true = halo_mass_particles[halo_mass_particles != 0]

    if log_m_bins is None:
        log_m_bins = np.arange(log_M_min, log_M_max, delta_log_M)

    n_particles_true, bins = np.histogram(np.log10(h_mass_true), bins=log_m_bins)

    particle_mass = hmf.get_particle_mass_correct_units(initial_parameters, with_h=True)
    total_mass = n_particles_true * particle_mass

    m_mid_bin = 10**((bins[1:] + bins[:-1])/2)
    return m_mid_bin, total_mass


def get_simulation_number_of_halos_from_particles_halo_mass(initial_parameters, log_M_min=10, log_M_max=15,
                                                            delta_log_M=0.1, log_m_bins=None):
    m_mid_bin, total_mass = get_simulation_total_mass(initial_parameters, log_M_min=log_M_min, log_M_max=log_M_max,
                                                      delta_log_M=delta_log_M, log_m_bins=log_m_bins)
    n_true_halos = hmf.get_number_halos_from_bins_total_mass(total_mass, np.log10(m_mid_bin))
    return m_mid_bin, n_true_halos


def num_den_simulation_from_particles_halo_mass(initial_parameters, log_M_min=10, log_M_max=15, delta_log_M=0.1,
                                                log_m_bins=None):
    m_mid_bin, n_true_halos = get_simulation_number_of_halos_from_particles_halo_mass(initial_parameters,
                                                                                      log_M_min=10, log_M_max=15,
                                                            delta_log_M=0.1, log_m_bins=None)
    n_density = hmf.get_comoving_number_density_halos(n_true_halos, initial_parameters)
    return m_mid_bin, n_density


def get_true_number_halos_per_mass_bins(initial_parameters, log_M_min=10, log_M_max=15, delta_log_M=0.1,
                                        log_m_bins=None, return_bins=False):
    if log_m_bins is None:
        log_m_bins = np.arange(log_M_min, log_M_max, delta_log_M)

    mass_halos = [initial_parameters.halo[i]['mass'].sum() for i in range(len(initial_parameters.halo))]
    mass_halos_h = np.array(mass_halos) * initial_parameters.final_snapshot.properties['h']

    n_halos, bins = np.histogram(np.log10(mass_halos_h), log_m_bins)
    if return_bins is True:
        return 10**bins, n_halos
    else:
        mid_bins = (bins[1:] + bins[:-1])/2
        return 10**mid_bins, n_halos


def get_true_total_mass_halos_per_mass_bins(initial_parameters, log_bins):
    mid_mass, n_halos = get_true_number_halos_per_mass_bins(initial_parameters, log_m_bins=log_bins, return_bins=False)

    total_mass = n_halos * mid_mass
    return mid_mass, total_mass


def get_f_nu_simulation(initial_parameters, variance_function, mass_bins=None, error=False):
    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")

    if mass_bins is None:
        mass_bins = np.logspace(10, 15, num=40)

    r_bins = (3 * mass_bins / (4 * np.pi * rho_M)) ** (1 / 3)
    k_bins = 2 * np.pi / r_bins
    var = variance_function(k_bins)

    ln_sig_sim = np.log(1 / np.sqrt(var))
    d_ln_sig_sim = np.diff(ln_sig_sim)

    m, n_halos = get_true_number_halos_per_mass_bins(initial_parameters, log_m_bins=np.log10(mass_bins))
    f_nu = m / rho_M * n_halos / d_ln_sig_sim / initial_parameters.boxsize_comoving ** 3

    if error is True:
        err = m / rho_M * np.sqrt(n_halos) / d_ln_sig_sim / initial_parameters.boxsize_comoving ** 3
        return var, f_nu, err
    else:
        return var, f_nu


def get_true_dndm(initial_parameters, log_mass_bins):
    m_mid, n_true = get_true_number_halos_per_mass_bins(initial_parameters, log_m_bins=log_mass_bins)
    n_den_true = hmf.get_comoving_number_density_halos(n_true, initial_parameters)
    dndm = hmf.get_dndm(n_den_true, 10**log_mass_bins)

    m_mid = (m_mid).view(pynbody.array.SimArray)
    m_mid.units = "Msol h^-1"
    return m_mid, dndm


def get_true_dndlog10m(initial_parameters, log_mass_bins):
    m_mid, n_true = get_true_number_halos_per_mass_bins(initial_parameters, log_m_bins=log_mass_bins)

    m_mid =(10**((log_mass_bins[1:] + log_mass_bins[:-1])/2)).view(pynbody.array.SimArray)
    m_mid.units = "Msol h^-1"

    n_den_true = hmf.get_comoving_number_density_halos(n_true, initial_parameters)
    dndlog_true = hmf.get_dn_dlog10m(n_den_true, 10 ** log_mass_bins, m_mid)

    return m_mid, dndlog_true