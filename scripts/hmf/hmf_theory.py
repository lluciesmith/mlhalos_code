"""
Functions regarding the theoretical halo mass function

"""
import numpy as np
import pynbody
from scripts.ellipsoidal import ellipsoidal_barrier as eb


def f_press_schechter(nu):
    return pynbody.analysis.hmf.f_press_schechter(nu)


def f_sheth_tormen(nu, Anorm=0.3222, a=0.707, p=0.3):
    return pynbody.analysis.hmf.f_sheth_tormen(nu, Anorm, a, p)


def get_nu_f_nu_theoretical(nu, kernel, Anorm=0.3222, a=0.707, p=0.3):
    if kernel == "ST":
        f = f_sheth_tormen(nu, Anorm=Anorm, a=a, p=p)
    elif kernel == "PS":
        f = f_press_schechter(nu)
    else:
        raise NameError("Kernel invalid - select either ST or PS")
    return f


def hmf_theory(initial_parameters, z=0, cosmology="WMAP5", kernel="ST", log_M_min=10.0, log_M_max=15.0, delta_log_M=0.1):
    if z == 0:
        snapshot = initial_parameters.final_snapshot
    elif z == 99:
        snapshot = initial_parameters.initial_conditions
    else:
        snapshot = None
        print("Not correct snapshot")

    pspec = eb.get_power_spectrum(cosmology, initial_parameters, z=z)
    M, sigma, N = pynbody.analysis.hmf.halo_mass_function(snapshot,
                                                          log_M_min=log_M_min, log_M_max=log_M_max,
                                                          delta_log_M=delta_log_M,
                                                          kern=kernel, pspec=pspec,
                                                          delta_crit=1.686)
    return M, sigma, N


def get_dndm_theory(initial_parameters, kernel="PS", cosmology="WMAP5", log_M_min=8, log_M_max=16, delta_log_M=0.1):
    m, sig, dndlog10_theory = hmf_theory(initial_parameters, z=0, cosmology=cosmology, kernel=kernel,
                                         log_M_min=log_M_min, log_M_max=log_M_max, delta_log_M=delta_log_M)

    dndm_theory = dndlog10_theory / (np.log(10) * m)
    return m, dndm_theory


def theoretical_number_halos(initial_parameters, kernel="PS", cosmology="WMAP5", log_M_min=10, log_M_max=15,
                             delta_log_M=0.1):
    m, dndm = get_dndm_theory(initial_parameters, kernel=kernel, cosmology=cosmology,log_M_min=log_M_min, log_M_max=log_M_max,
                              delta_log_M=delta_log_M)
    boxsize_comoving = initial_parameters.boxsize_comoving
    volume = boxsize_comoving ** 3
    m_bins = 10 ** np.arange(log_M_min, log_M_max, delta_log_M)
    delta_m = np.diff(m_bins)
    num = dndm * delta_m * volume
    return m, num


def hmf_theory_Nina_Pk(initial_parameters, z=0, cosmology="WMAP5", kernel="ST", filter="top hat",
               log_M_min=10.0, log_M_max=15.0, delta_log_M=0.1):
    if z == 0:
        snapshot = initial_parameters.final_snapshot
    elif z == 99:
        snapshot = initial_parameters.initial_conditions
    else:
        snapshot = None
        print("Not correct snapshot")

    pspec_Nina = pynbody.analysis.hmf.PowerSpectrumCAMB(initial_parameters.final_snapshot,
                                                        filename="/Users/lls/Desktop/nina_camb_Pk_WMAP5")
    M, sigma, N = pynbody.analysis.hmf.halo_mass_function(snapshot,
                                                          log_M_min=log_M_min, log_M_max=log_M_max,
                                                          delta_log_M=delta_log_M,
                                                          kern=kernel, pspec=pspec_Nina,
                                                          delta_crit=1.686)
    return M, sigma, N