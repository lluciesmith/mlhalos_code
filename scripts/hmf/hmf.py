#matplotlib.use("macosx")
import sys
import numpy as np
sys.path.append("/Users/lls/Documents/mlhalos_code")
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from scripts.hmf import predict_masses as mp
import pynbody
import importlib
importlib.reload(mp)
importlib.reload(eb)


def get_mid_mass_bins(bins, units=None):
    mid_bins = (bins[1:] + bins[:-1])/2
    mid_bins = pynbody.array.SimArray(mid_bins)
    if units is None:
        mid_bins.units = bins.units
    else:
        mid_bins.units = units
    return mid_bins


def get_log_bin_masses_and_mid_values(initial_parameters=None, window_parameters=None, mass_bins_type="smoothing",
                                      num_linspace=None, delta_log_M=None):
    """Returns log-scale masses in units [Msol h^-1] """
    if mass_bins_type == "smoothing":
        h = initial_parameters.initial_conditions.properties['h']

        m = window_parameters.smoothing_masses * h
        m.units = "Msol h^-1"
    else:
        if num_linspace is None:
            num_linspace = 50
        n, m = np.histogram(mass_bins_type, bins=num_linspace)
        m = m.view(pynbody.array.SimArray)
        m.units = mass_bins_type.units

    assert m.units == "Msol h^-1"

    log_M = np.log10(m)
    log_M_min = log_M.min()
    log_M_max = log_M.max()
    if delta_log_M is None:
        delta_log_M = log_M[2] - log_M[1]

    M_mid = np.linspace(log_M_min + delta_log_M / 2, log_M_max - delta_log_M / 2, num=len(log_M)-1, endpoint=True)
    M_mid = pynbody.array.SimArray(M_mid)
    M_mid.units = log_M.units
    return log_M, M_mid


def get_particle_mass_correct_units(initial_parameters, with_h=True):
    if with_h is True:
        mass_in_correct_units = initial_parameters.initial_conditions['mass'].in_units("Msol h^-1")
    else:
        mass_in_correct_units = initial_parameters.initial_conditions['mass']
    particle_mass = pynbody.array.SimArray(mass_in_correct_units[0])
    particle_mass.units = mass_in_correct_units.units
    return particle_mass


def get_number_halos_from_bins_total_mass(total_mass, log_M_middle):
    m_mid = 10 ** log_M_middle
    m_mid.units = log_M_middle.units
    number_halos = total_mass / m_mid
    return number_halos


def get_comoving_number_density_halos(number_halos, initial_parameters):
    boxsize = initial_parameters.boxsize_comoving
    boxsize = pynbody.array.SimArray([boxsize])
    boxsize.units = "Mpc h**-1 a"
    volume = boxsize**3
    number_density = number_halos / volume
    return number_density


def get_dndm(n, m):
    dndm = n / np.diff(m)
    return dndm


def get_dn_dlog10m(n, m, m_mid):
    dndm = get_dndm(n, m)
    dndlnm = m_mid * dndm
    dndlog10m = np.log(10) * dndlnm
    return dndlog10m