"""
How does the theoretical mass function or the Sheth-Tormen collapse barrier change
as we change the definition of mass variance? ( Or, in other words, as we smooth
it with a different window function? )

"""
import sys

import scripts.ellipsoidal.power_spectrum

sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from mlhalos import parameters
import pynbody
import matplotlib.pyplot as plt
from scripts.hmf import hmf
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy
from scripts.hmf.larger_sim import hmf_analysis as ha


def top_hat_variance(k_scales, initial_parameters, z=0, cosmology="WMAP5"):
    r_th = pynbody.array.SimArray(2*np.pi/k_scales)
    r_th.sim = initial_parameters.initial_conditions
    r_th.units = "Mpc a h**-1"
    v_th = eb.calculate_variance(r_th, initial_parameters, z=z, cosmology=cosmology)
    return r_th, v_th


def top_hat_mass(k_scales, initial_parameters):
    r_th = pynbody.array.SimArray(2 * np.pi / k_scales)
    m_th = initial_parameters.mean_density * 4/3 * np.pi * (r_th * 0.01 / 0.701)**3 * 0.701
    m_th = pynbody.array.SimArray(m_th)
    m_th.units = "Msol h^-1"
    return m_th


def sharp_k_variance(scales, initial_parameters, z=0, cosmology="WMAP5"):
    if scales.units == "Mpc^-1 a^-1 h":
        print("You have k")
        r_sk = pynbody.array.SimArray(1 / scales)
        r_sk.sim = initial_parameters.initial_conditions
        r_sk.units = "Mpc a h**-1"
    elif scales.units == "Mpc a h^-1":
        print("You have r")
        r_sk = scales
    else:
        raise(NameError, "Not the correct units")
    SK = eb.SharpKFilter(initial_parameters.initial_conditions)
    v_sk = eb.calculate_variance(r_sk, initial_parameters, filter=SK, z=z, cosmology=cosmology)

    return r_sk, v_sk


def default_sk_mass(k_scales, initial_parameters):
    r_sk = pynbody.array.SimArray(1 / k_scales)
    m_sk = initial_parameters.mean_density * 6 * np.pi**2 * (r_sk * 0.01 / 0.701)**3 * 0.701
    m_sk = pynbody.array.SimArray(m_sk)
    m_sk.units = "Msol h^-1"
    return m_sk


# def mass_sk(radius, param):
#     rho_M = pynbody.analysis.cosmology.rho_M(ic_small.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
#     return rho_M * 4/3 * np.pi * (param * radius)**3

def mass_sk(mass_top_hat, param):
    return param**3 * mass_top_hat


def plot_variance_vs_k_and_mass(initial_parameters):
    L = initial_parameters.boxsize_comoving
    k_min = 2*np.pi/L
    k_nyquist = k_min * np.sqrt(3) * initial_parameters.shape/2
    k_max = k_nyquist/2
    k = np.logspace(np.log10(k_min*2), np.log10(k_max), num=50, endpoint=True)

    # top-hat variance and mass

    m_th = top_hat_mass(k, initial_parameters)
    v_th = top_hat_variance(k, initial_parameters)

    # Sharp-k variance and mass

    v_sk = sharp_k_variance(k, initial_parameters)
    m_sk = default_sk_mass(k, initial_parameters)

    #m_sk_2 = ic.mean_density * 4/3 * np.pi * (r_sk * 0.01 / 0.701)**3 * 0.701

    f, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 5.2))

    # ax1.loglog(r_sk, v_sk, label="sharp-k", color="b")
    # ax1.loglog(r_th, v_th, label="top-hat", color="g")
    # ax1.legend(loc="best", frameon=False)
    # ax1.set_xlabel("Smoothing radius")
    # ax1.set_ylabel(r"$ \sigma^2 $")

    ax2.loglog(k/k_min, v_sk, label="sharp-k", color="b")
    ax2.loglog(k/k_min, v_th, label="top-hat", color="g")
    ax2.legend(loc="best", frameon=False)
    ax2.set_xlabel(r"$k / (2 \pi / L)$")
    ax2.set_ylabel(r"$ \sigma^2 $")

    plt.subplots_adjust(wspace=0.05)

    ax3.loglog(m_sk, v_sk, label="sharp-k", color="b")
    #ax3.loglog(m_sk_2, v_sk, label="sharp-k (V sphere)", color="grey")
    ax3.loglog(m_th, v_th, label="top-hat", color="g")
    ax3.legend(loc="best", frameon=False)
    ax3.set_xlabel(r"$\mathrm{Mass} [ M_{\odot}/h ]$")
    ax3.set_yticklabels(ax2.get_yticklabels(), visible=False)


def hmf_theory_variance_choice(initial_parameters, variance="top hat", kernel="PS", log_M_min=10.0, log_M_max=15.0,
                                                          delta_log_M=0.1):
    pspec = scripts.ellipsoidal.power_spectrum.get_power_spectrum("WMAP5", initial_parameters, z=0)
    if variance == "sharp-k":
        pspec._default_filter = eb.SharpKFilter(initial_parameters.final_snapshot)

    M, sigma, N = pynbody.analysis.hmf.halo_mass_function(initial_parameters.final_snapshot, pspec=pspec,
                                                          kern=kernel, log_M_min=log_M_min, log_M_max=log_M_max,
                                                          delta_log_M=delta_log_M,
                                                          delta_crit=1.686)
    volume = initial_parameters.boxsize_comoving ** 3
    m_bins = 10 ** np.arange(log_M_min, log_M_max, delta_log_M)
    delta_m = np.diff(m_bins)
    num_halos = N * delta_m * volume / (np.log(10) * M)
    return M, sigma, num_halos


    # Do the same for the large box
    #
    # ic_large = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/sim200/sim200.gadget3",
    #                                             final_snapshot="/Users/lls/Documents/CODE/standard200/snapshot_011",
    #                                             path="/Users/lls/Documents/CODE/")

if __name__ == "__main":
    ic_small = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")


    plot_variance_vs_k_and_mass(ic_small)

# Fit radius relation


def sharp_k_variance_radius_input(scales, initial_parameters):
    r_sk = scales
    SK = eb.SharpKFilter(initial_parameters.initial_conditions)
    v_sk = eb.calculate_variance_radius(r_sk, initial_parameters, filter=SK)

    return r_sk, v_sk


def fit(initial_parameters):
    L = initial_parameters.boxsize_comoving
    k_min = 2*np.pi/L
    k_nyquist = k_min * np.sqrt(3) * initial_parameters.shape/2
    k_max = k_nyquist/2
    k = np.logspace(np.log10(k_min*2), np.log10(k_max*2), num=50, endpoint=True)

    r_th, v_th = top_hat_variance(k, initial_parameters)
    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    m_th = rho_M * 4 / 3 * np.pi * r_th ** 3

    # k = pynbody.array.SimArray(k)
    # k.units = "Mpc^-1 a^-1 h"
    #
    # rho_M = pynbody.analysis.cosmology.rho_M(ic_small.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    # r_th, v_th = top_hat_variance(k, initial_parameters)
    # m_th = rho_M * 4 / 3 * np.pi * r_th ** 3
    # m_sk = rho_M * 4 / 3 * np.pi * r_sk ** 3 * alpha(r_sk)

    def sigma_2_m_sk(log_m, alpha):
        m = 10**log_m
        r_sk = (m * 3 / (4 * np.pi * rho_M * alpha))**(1/3)
        #r_sk = pynbody.array.SimArray(r_sk)
        #r_sk.units = "Mpc a h^-1"
        return np.log10(sharp_k_variance_radius_input(r_sk, initial_parameters)[1])

    popt = np.zeros((len(m_th),))
    pcov = np.zeros((len(m_th),))
    for i in range(len(m_th)):
        m_i = m_th[i]
        v_i = v_th[i]

        popt[i], pcov[i] = curve_fit(sigma_2_m_sk, np.log10(m_i), np.log10(v_i))

    plt.plot(np.log10(m_th), np.log10(v_th), label="data")
    plt.plot(np.log10(m_th), sigma_2_m_sk(np.log10(m_th), popt), label="fit")
    plt.legend(loc="best")

    #r_sk = popt ** (1 / 3) * r_th
    r_sk = r_th * (popt**(1/3))
    f = scipy.interpolate.interp1d(r_th, r_sk)
    m_sk = 4 / 3 * np.pi * f(r_th) ** 3 * rho_M
    fm = scipy.interpolate.interp1d(m_th, m_sk)

    pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                             "ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")

    pred_rescaled = np.zeros((len(pred_spherical),))
    #ind = np.where(~np.isnan(pred_spherical) & (pred_spherical >= (3 * 10**10)))[0]
    ind = np.where(~np.isnan(pred_spherical))[0]
    pred_rescaled[ind] = fm(pred_spherical[ind])
    pred_rescaled[np.isnan(pred_spherical)] = pred_spherical[np.isnan(pred_spherical)]

    m_min = 10
    m_max = 15
    delta_log_M = 0.1
    kernel = "PS"
    color = "b"
    volume = "small"

    m_sph, num_sph = ha.get_empirical_number_density_halos(pred_spherical, initial_parameters, boxsize=L,
                                                           log_M_min=m_min, log_M_max=m_max, delta_log_M=delta_log_M)
    m_sk_rescaled, num_sk_rescaled = ha.get_empirical_number_density_halos(pred_rescaled, boxsize=L,
                                                                           initial_parameters=initial_parameters,
                                                                           log_M_min=m_min,
                                                                           log_M_max=m_max, delta_log_M=delta_log_M)
    m_theory, num_theory = ha.get_theory_number_density_halos(kernel, initial_parameters, boxsize=L, log_M_min=m_min,
                                                              log_M_max=m_max, delta_log_M=delta_log_M)

    pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                      "PS_predicted_mass_100_scales_extended_low_mass_range.npy")

    m_sk, num_sk = ha.get_empirical_number_density_halos(pred_sk, boxsize=boxsize, initial_parameters=ic,
                                                         log_M_min=m_min, log_M_max=m_max, delta_log_M=delta_log_M)
    m_bins = np.arange(m_min, m_max, delta_log_M)
    bins = 10 ** m_bins
    delta_m = np.diff(bins)

    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    poisson_ps = [np.random.poisson(num_i, 10000) for num_i in num_theory * V]
    vplot2 = ax2.violinplot(poisson_ps, positions=m_theory, widths=delta_m, showextrema=False, showmeans=False, showmedians=False)

    ax2.step(bins[:-1], num_theory * V, where="post", color=color)
    ax2.plot([bins[-2], bins[-1]], [num_theory[-1] * V, num_theory[-1] * V], color=color)
    [b.set_color(color) for b in vplot2['bodies']]

    #ax2.plot(m_theory, num_theory * V, color="k", label="theory")

    ax2.scatter(m_sph, num_sph * L**3, label="volume sphere", marker="^",color=color)
    ax2.scatter(m_sk, num_sk * L**3, label="volume sharp-k", marker="o",color=color)
    # ax2.scatter(m_sph * (9 * np.pi /2), num_sph * V, label="horizontal shift only", marker="x", color="k")

    # ax2.scatter(m_sph * (9*np.pi/2), num_sph* V/ (9*np.pi/2), label="analytic expectation", marker="x")
    ax2.scatter(m_sk_rescaled, num_sk_rescaled * L**3 , label="shifted spherical predictions",color="k", marker="x")

    ax2.legend(loc="best", fontsize=15, frameon=False)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # ax2.set_xlim(10**11, 10**14)
    # ax2.set_ylim(10, 2 * 10**3)
    ax2.set_xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
    ax2.set_ylabel("Number of halos")

    #ax2.set_title(title + " (" + volume + " box)")

    m_min_plot = initial_parameters.initial_conditions['mass'].in_units("Msol h**-1")
    m_max_plot = (np.where(num_theory*L**3 >= 10)[0]).max()
    ax2.set_xlim(m_min_plot[0]*100, 10 ** (np.log10(m_theory[m_max_plot]) + (delta_log_M / 2)))
    ax2.set_ylim(1, 2*10**3)
    plt.tight_layout()












#     def mass_sk(r_sk, alpha):
#         return 4 / 3 * np.pi * r_sk ** 3 * alpha(r_sk)
#
#
#     m_th = top_hat_mass(k, initial_parameters)
#
#     def fit_variance_sk(mass_scales, param):
#         rho_M = pynbody.analysis.cosmology.rho_M(ic_small.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
#         radius = pynbody.array.SimArray((param * mass_scales / (rho_M * 4/3 * np.pi))**(1/3))
#         radius.units = "Mpc a h^-1"
#
#         SK = eb.SharpKFilter(initial_parameters.initial_conditions)
#         v_sk = eb.calculate_variance(radius, initial_parameters, filter=SK)
#         return v_sk
#
#     popt, pcov = curve_fit(fit_variance_sk, m_th, v_th)
#     return popt, pcov
#
#
# #### try something different ####
#
# initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
#
# m = 10**np.arange(np.log10(9.9*10**9), 15, 0.1)
# rho_M = pynbody.analysis.cosmology.rho_M(ic_small.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
# r_th = (m / (4/3 * np.pi * rho_M))**(1/3)
#
#
# def sigma_2_sharp_k(m_sk, alpha):
#     r_th = (m_sk / (4/3 * np.pi * rho_M))**(1/3)
#     r_sk = pynbody.array.SimArray(r_th * alpha)
#     r_sk.units = "Mpc a h^-1"
#     return sharp_k_variance(r_sk, initial_parameters)[1]
#
# def sigma_2_top_hat(m_th):
#     r_th = (m_th / (4/3 * np.pi * rho_M))**(1/3)
#     r_th = pynbody.array.SimArray(r_th)
#     r_th.units = "Mpc a h^-1"
#     return top_hat_variance(r_th, initial_parameters)[1]
#
# v_th = sigma_2_top_hat(m)
#
# popt, pcov = curve_fit(sigma_2_sharp_k, m, v_th)
# best_fit = sigma_2_sharp_k(m, popt)
#
# plt.loglog(m, v_th, label="top-hat")
# plt.loglog(m, best_fit, label="fit")
# plt.legend(loc="best")


