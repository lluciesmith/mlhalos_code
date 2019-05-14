import numpy as np
import sys
import scipy.integrate

import scripts.ellipsoidal.power_spectrum
camb_path = "/home/lls/stored_files/"
# camb_path = "/Users/lls/Software/CAMB-Jan2017/"
# mlhalos_path = "/Users/lls/Documents/mlhalos_code"
# sys.path.append(mlhalos_path)
sys.path.append(camb_path)
#import pycamb.camb as camb
import pynbody
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from mlhalos import parameters
import matplotlib.pyplot as plt


def get_camb_pk_example():
    # Now get matter power spectra and sigma8 at redshift 0 and 0.8

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=70.1, ombh2=0.022113044999999994, omch2=0.11498783399999998)
    pars.set_dark_energy() # re-set defaults
    pars.InitPower.set_params(ns=0.96)

    # Not non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)

    #Linear spectra
    #pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    #results.get_transfer_functions(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    s8 = np.array(results.get_sigma8())

    plt.loglog(kh, pk[0,:], color='k')
    plt.loglog(kh, pk[1,:], color='k', ls="--")


def get_ICs_power_spectrum(initial_parameters, z=0, output_file=None):
    if output_file is None:
        output_file = np.loadtxt('/Users/lls/Downloads/IC_doub_z99_256_0.ps')

    if z != 99:
        linear_growth = pynbody.analysis.cosmology.linear_growth_factor(initial_parameters.initial_conditions)
    else:
        linear_growth = 1

    kh = output_file[:, 0]
    pspec_theory_z0 = output_file[:, 2]/(linear_growth ** 2)
    pspec_realisation_z0 = output_file[:, 3]/(linear_growth ** 2)

    return kh, pspec_theory_z0, pspec_realisation_z0


def get_sigma_raw_powerspectrum(R, initial_parameters):
    """radius in [Mpc h_1 a] """
    snapshot = initial_parameters.final_snapshot
    th = pynbody.analysis.hmf.TophatFilter(snapshot)

    k, Pk, Pk_realisation = get_ICs_power_spectrum(initial_parameters)
    #
    # snapshot.properties['sigma8'] = 1
    # pspec_Nina = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot,
    #                                                         filename="/Users/lls/Desktop/nina_camb_Pk_WMAP5")
    # variance = int( k**2 * pk * W(kr)**2) / (2*np.pi**2)
    # Need P(k) as a callable function of k
    #

    dlnk = np.log(k[1] / k[0])

    # we multiply by k because our steps are in logk.

    integrand = Pk * (k ** 3) * (th.Wk(R*k) ** 2)
    sigma = (0.5 / np.pi ** 2) * scipy.integrate.simps(integrand, dx=dlnk, axis=-1)
    return np.sqrt(sigma)

    # integrand = lambda k: k ** 2 * pspec_Nina(k) * th.Wk(k * R) ** 2
    # integrand_ln_k = lambda k: np.exp(k) * integrand(np.exp(k))
    # v = scipy.integrate.romberg(integrand_ln_k, np.log(min(pspec_Nina.k)), np.log(
    #     1. / R) + 3, divmax=10, rtol=1.e-4) / (2 * np.pi ** 2)
    # return v


def get_my_WMAP5_k_pk_from_camb():
    pspec = scripts.ellipsoidal.power_spectrum.get_power_spectrum("WMAP5", ic, z=0)
    return pspec


def hmf_theory_Nina_Pk(initial_parameters, z=0, cosmology="WMAP5", kernel="ST", filter="top hat",
               log_M_min=10.0, log_M_max=15.0, delta_log_M=0.1):
    if z == 0:
        snapshot = initial_parameters.final_snapshot
    elif z == 100:
        snapshot = initial_parameters.initial_conditions
    else:
        snapshot = None
        print("Not correct snapshot")

    pspec_Nina = pynbody.analysis.hmf.PowerSpectrumCAMB(ic.final_snapshot,
                                                        filename="/Users/lls/Desktop/nina_camb_Pk_WMAP5")
    # if filter == "sharp k":
    #     pspec._default_filter = pynbody.analysis.hmf.HarmonicStepFilter(snapshot)
    M, sigma, N = pynbody.analysis.hmf.halo_mass_function(snapshot,
                                                          log_M_min=log_M_min, log_M_max=log_M_max,
                                                          delta_log_M=delta_log_M,
                                                          kern=kernel, pspec=pspec_Nina,
                                                          delta_crit=1.686)
    return M, sigma, N

if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    k, pk_theory, pk_real = get_ICs_power_spectrum(ic, z=0, output_file=None)

    pspec_Nina = np.column_stack((k, pk_theory))
    np.savetxt("/Users/lls/Desktop/nina_camb_Pk_WMAP5", pspec_Nina)

    powerspec_Nine = pynbody.analysis.hmf.PowerSpectrumCAMB(ic.final_snapshot,
                                                            filename="/Users/lls/Desktop/nina_camb_Pk_WMAP5")

    pwspectrum = get_my_WMAP5_k_pk_from_camb()

    print(pwspectrum.get_sigma8())

    plt.loglog(pwspectrum.k, pwspectrum(pwspectrum.k) / (2 * np.pi) ** 3, label="my Pk")
    plt.loglog(k, pk_theory, label="ICs Pk theory")
    plt.loglog(k, pk_real, label="ICs Pk realisation")
    plt.legend(loc="best")
    plt.xlim(10 ** -4, 10 ** 2)
    plt.ylim(10 ** -5, 10 ** 5)
    plt.xlabel("k [h/Mpc]")
    plt.ylabel("P(k)")
    # plt.savefig("/Users/lls/Desktop/power_spectrum_realisation_vs_sim.png")





