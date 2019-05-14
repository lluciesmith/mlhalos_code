import numpy as np
import pynbody

from scripts.hmf import hmf_simulation as hmf_sim
from scripts.hmf import hmf_empirical as hmf_emp
import scripts.hmf.hmf_theory
from mlhalos import parameters
from scripts.ellipsoidal import variance_analysis as va
from scripts.ellipsoidal import ellipsoidal_barrier as eb
import matplotlib.pyplot as plt
import scipy.interpolate


def get_variance_function(mass, initial_parameters, th=True):
    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    r = (3 * mass / (4 * np.pi * rho_M)) ** (1 / 3)
    k = pynbody.array.SimArray(2 * np.pi / r)
    k.units = "Mpc^-1 a^-1 h"
    if th is True:
        r1, var = va.top_hat_variance(k, initial_parameters, z=0)
    else:
        r1, var = va.sharp_k_variance(k, initial_parameters, z=0)
    f = scipy.interpolate.interp1d(k, var)
    return f


def get_variance_as_function_mass(mass, initial_parameters):
    mass = mass.view(pynbody.array.SimArray)
    mass.units = "Msol h^-1"
    v = eb.calculate_variance(mass, initial_parameters, z=0)
    return v


#####################  Initial parameters + variance interpolation  ####################

# large sim

initial_snapshot = "/Users/lls/Documents/CODE/standard200/standard200.gadget3"
final_snapshot = "/Users/lls/Documents/CODE/standard200/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                final_snapshot=final_snapshot, path="/Users/lls/Documents/CODE/")

# small sim

ic_small = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
rho_M = pynbody.analysis.cosmology.rho_M(ic_small.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")

# variance interpolation

mass_scales_variance = 10**np.arange(8, 16, 0.1)
var_function = get_variance_function(mass_scales_variance, ic_small)



# m_plot = 10**np.arange(10.5, 15, 0.1)
# r_plot = (3 * m_plot / (4 * np.pi * rho_M)) ** (1 / 3)
# k_plot = 2 * np.pi / r_plot
# fig, ax1 = plt.subplots()
# ax2 = ax1.twiny()
# ax1.loglog(m_plot, np.sqrt(var_function(k_plot)))
# ax2.set_xlabel(r"$\log [1/ \sigma ]$", color="r")
#
# ax2.set_xticks(new_tick_locations)
# ax2.set_xticklabels(tick_function(new_tick_locations))
# ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
#
# ax2.tick_params(np.log(1/np.sqrt(var_function(k_plot))), colors='r')
# plt.legend(loc="best")
# ax1.set_xlabel("mass")
# ax1.set_ylabel(r"$\sigma$")
# plt.tight_layout()



#####################  simulation  ####################

m = 10**np.arange(10.5, 14, 0.1)
variance_small, f_small, err_small = hmf_sim.get_f_nu_simulation(ic_small, var_function, mass_bins=m, error=True)
v_small = (variance_small[1:] + variance_small[:-1])/2

m2 = 10**np.arange(11.5, 15.5, 0.1)
variance_large, f_large, err_large = hmf_sim.get_f_nu_simulation(ic_200, var_function, mass_bins=m2, error=True)
v_large = (variance_large[1:] + variance_large[:-1])/2


#################### theory ####################

delta_sc = eb.get_spherical_collapse_barrier(ic_small, z=0, delta_sc_0=1.686, output="delta")

v_th = np.arange(0.05, 10, step=0.01)
nu = delta_sc / np.sqrt(v_th)
f_PS = scripts.hmf.hmf_theory.get_nu_f_nu_theoretical(nu, "PS")
f_ST = scripts.hmf.hmf_theory.get_nu_f_nu_theoretical(nu, "ST")


# PLOT THEORY VS SIMULATION

var_tick = get_variance_as_function_mass(mass_scales_variance, ic_small)
x = np.log(1/np.sqrt(var_tick))
tick_function = scipy.interpolate.interp1d(x, mass_scales_variance)

def xticklab(var):
    return ["%.2f" % z for z in np.log10(tick_function(var))]


fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twiny()
ax1.plot(np.log(nu/delta_sc), np.log(f_PS), label="PS")
ax1.plot(np.log(nu/delta_sc), np.log(f_ST), label="ST")
ax1.errorbar(np.log(1/np.sqrt(v_small)), np.log(f_small), yerr=err_small/f_small,
             fmt="^", color="k", label="sim(small)", markersize=5)
ax1.errorbar(np.log(1/np.sqrt(v_large)), np.log(f_large), yerr=err_large/f_large,
             fmt="x", color="k", label="sim(large)", markersize=5)

ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(xticklab(ax1.get_xticks()), fontsize=15)
ax2.set_xlabel(r"$\log(M/M_{\odot})$")
#ax2.ticklabel_format(style='sci')

ax1.legend(loc="best", frameon=False)
ax1.set_xlabel(r"$\log [1/ \sigma ]$")
ax1.set_ylabel(r"$\log [f(\nu)]$")
ax1.set_ylim(-5, 0)
fig.subplots_adjust(top=0.88)


####################  emp  ####################

# from k predictions


def get_f_nu_empirical(number_particles, k_bins, initial_parameters):
    ks = k_bins.view(pynbody.array.SimArray)
    ks.units = "Mpc^-1 a^-1 h"

    bins_r, bins_variance = va.sharp_k_variance(ks, initial_parameters, z=0)
    ln_sig = np.log(1/np.sqrt(bins_variance))
    d_ln_sig = np.diff(ln_sig)
    m_particle = initial_parameters.initial_conditions['mass'][0] * \
                 initial_parameters.initial_conditions.properties['h']

    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    f_emp = number_particles * m_particle / (rho_M * initial_parameters.boxsize_comoving ** 3) / abs(d_ln_sig)
    mid_sig = (ln_sig[1:] + ln_sig[:-1]) / 2
    return mid_sig, f_emp


def get_error_empirical(predictions, log_k_bins, initial_parameters, mean_density):
    k_bins = (10 ** log_k_bins).view(pynbody.array.SimArray)
    k_bins.units = "Mpc^-1 a^-1 h"

    r_bins = 1 / k_bins
    m_bins = mean_density * 4 / 3 * np.pi * r_bins ** 3

    num_particles_predicted, b = np.histogram(predictions[~np.isnan(predictions)], m_bins[::-1])
    m, num_halos = hmf_emp.get_number_halos_from_number_particles_binned(num_particles_predicted, np.log10(b),
                                                                       initial_parameters)

    bins_r, bins_variance = va.sharp_k_variance(k_bins, initial_parameters, z=0)
    ln_sig = np.log(1/np.sqrt(bins_variance))
    d_ln_sig = np.diff(ln_sig)
    m_particle = initial_parameters.initial_conditions['mass'][0] * \
                 initial_parameters.initial_conditions.properties['h']

    err = m_particle / mean_density * np.sqrt(num_halos[::-1]) / abs(d_ln_sig[::-1]) / \
          initial_parameters.boxsize_comoving ** 3
    return err


def get_f_nu_from_k_predictions(predictions, initial_parameters, log_k_bins=30):
    num_k, bins_k = np.histogram(np.log10(predictions[~np.isnan(predictions)]), log_k_bins)

    mid_sig, f_emp = get_f_nu_empirical(num_k, 10**bins_k, initial_parameters)
    return mid_sig, f_emp, bins_k


def get_f_nu_from_spherical_prediction(predicted_masses, initial_parameters, rho_M, log_k_bins=30):
    r_predicted = (predicted_masses / (rho_M * 4/3 * np.pi))**(1/3)
    k_predicted = 1/r_predicted

    num_particles, k_bins = np.histogram(np.log10(k_predicted[~np.isnan(k_predicted)]), log_k_bins)
    mid_sig, f_nu = get_f_nu_empirical(num_particles, 10**k_bins, initial_parameters)
    #num_particles, k_bins = np.histogram(k_predicted[~np.isnan(k_predicted)], log_k_bins)
    #mid_sig, f_nu = get_f_nu_empirical(num_particles, k_bins, initial_parameters)
    return mid_sig, f_nu, k_bins


def get_f_nu_from_sk_prediction(predicted_masses, initial_parameters, rho_M, log_k_bins=30):
    k_predicted = (rho_M * 6 * np.pi**2 / predicted_masses) ** (1/3)

    num_particles, k_bins = np.histogram(np.log10(k_predicted[~np.isnan(k_predicted)]), log_k_bins)
    mid_sig, f_nu = get_f_nu_empirical(num_particles, 10**k_bins, initial_parameters)
    return mid_sig, f_nu, k_bins

def get_f_nu_from_radii_prediction(predicted_radii, initial_parameters, log_k_bins=30):
    k_predicted = 1/predicted_radii

    num_particles, k_bins = np.histogram(np.log10(k_predicted[~np.isnan(k_predicted)]), log_k_bins)
    mid_sig, f_nu = get_f_nu_empirical(num_particles, 10**k_bins, initial_parameters)
    return mid_sig, f_nu, k_bins


#################### PS ####################

# small box
# pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/"
#                         "ALL_predicted_masses_1500_even_log_m_spaced.npy")
# sig_sph, f_sph, lkb = get_f_nu_from_spherical_prediction(pred_spherical, ic_small, rho_M, log_k_bins=30)
# err_sph = get_error_empirical(pred_spherical, lkb, ic_small, rho_M)

pred_spherical_growth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                        "ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
sig_sph_growth, f_sph_growth, lkb_growth = get_f_nu_from_spherical_prediction(pred_spherical_growth, ic_small, rho_M,
                                                                              log_k_bins=30)
err_sph_growth = get_error_empirical(pred_spherical_growth, lkb_growth, ic_small, rho_M)


pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                  "PS_predicted_mass_100_scales_extended_low_mass_range.npy")
mid_sig_sk, f_nu_sk, lkb_sk = get_f_nu_from_sk_prediction(pred_sk, ic_small, rho_M, log_k_bins=lkb_growth)

PS_k_predicted = np.load("/Users/lls/Documents/CODE/stored_files/hmf/upcrossings/small_box/PS_k_predicted.npy")
mig_sig_k, f_nu_k, lbk_k = get_f_nu_from_k_predictions(PS_k_predicted, ic_small, log_k_bins=lkb_growth)


# big box
pred_spherical_large_PS= np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")
large_PS_001 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/growth_001/ALL_PS_predicted_radii.npy")
sig_sph_large, f_sph_large, lkb_large = get_f_nu_from_spherical_prediction(pred_spherical_large_PS, ic_200, rho_M,
                                                                     log_k_bins=30)
sig_sph_large_001, f_sph_large_001, lkb_large_001 = get_f_nu_from_radii_prediction(large_PS_001, ic_200,
                                                                     log_k_bins=30)

# pred_sk_large_PS= np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/volume_sharp_k/ALL_PS_predicted_masses"
#                            ".npy")
# m_sig_sk_large_PS, f_nu_sk_large_PS, lkb = get_f_nu_from_sk_prediction(pred_sk_large_PS, ic_200, rho_M,
#                                                                       log_k_bins=lkb_large)

plt.plot(np.log(nu/delta_sc), np.log(f_PS), label="PS (theory)", color="k")
plt.plot(sig_sph_growth, np.log(f_sph_growth), label="small")
plt.plot(sig_sph_large, np.log(f_sph_large), label="large")
#plt.plot(mig_sig_k, np.log(f_nu_k), label="k-pred")
#plt.plot(mid_sig_sk, np.log(f_nu_sk), label="sk")
plt.legend(loc="best", frameon=False)
plt.xlabel(r"$\log [1/ \sigma ]$")
plt.ylabel(r"$\log [f(\nu)]$")
plt.ylim(-5, 0)
plt.xlim(-1.3, 1.2)
plt.title("Press-Schechter")
plt.tight_layout()

plt.plot(np.log(1/np.sqrt(v_th)), np.log(f_PS), label="PS (theory)", color="k")
plt.plot(sig_sph_large, np.log(f_sph_large), label="correct growth")
# plt.plot(sig_sph_large, np.log(f_sph_large), label="spherical(large)")
plt.plot(sig_sph_large_001, np.log(f_sph_large_001), label=r"D(a)=a")
plt.xlabel(r"$\log [1/ \sigma ]$")
plt.ylabel(r"$\log [f(\nu)]$")
plt.ylim(-5, 0)
#plt.xlim(-1.3, 1.2)
plt.legend()
plt.subplots_adjust(top=0.88)
plt.title("Press-Schechter (large box)")
#plt.tight_layout()



fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twiny()
ax1.plot(np.log(nu/delta_sc), np.log(f_PS), label="PS", color="k", ls="--")
ax1.errorbar(sig_sph, np.log(f_sph), yerr=err_sph/f_sph, label=r" PS $D(a) = a$")
ax1.errorbar(sig_sph_growth, np.log(f_sph_growth), yerr=err_sph_growth/f_sph_growth, label=r"correct $D(a)$")

ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(xticklab(ax1.get_xticks()), fontsize=15)
ax2.set_xlabel(r"$\log(M)$")
#ax2.ticklabel_format(style='sci')

ax1.legend(loc="best", frameon=False)
ax1.set_xlabel(r"$\log [1/ \sigma ]$")
ax1.set_ylabel(r"$\log [f(\nu)]$")
ax1.set_ylim(-3, 0)
fig.subplots_adjust(top=0.88)


fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twiny()
ax1.plot(np.log(nu/delta_sc), np.log(f_PS), label="PS")
ax1.errorbar(sig_sph, np.log(f_sph), yerr=err_small/f_sph, label="spherical(small)")

ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(xticklab(ax1.get_xticks()), fontsize=15)
ax2.set_xlabel(r"$\log(M)$")
#ax2.ticklabel_format(style='sci')

ax1.legend(loc="best", frameon=False)
ax1.set_xlabel(r"$\log [1/ \sigma ]$")
ax1.set_ylabel(r"$\log [f(\nu)]$")
ax1.set_ylim(-5, 0)
fig.subplots_adjust(top=0.88)

#################### ST ####################

ST_pred_spherical_growth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                            "ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
ST_sig_sph_growth, ST_f_sph_growth, ST_lkb_growth = get_f_nu_from_spherical_prediction(ST_pred_spherical_growth,
                                                                                      ic_small,
                                                                                       rho_M, log_k_bins=30)

ST_pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/"
                            "ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
ST_sig_sph, ST_f_sph, ST_lkb = get_f_nu_from_spherical_prediction(ST_pred_spherical, ic_small, rho_M,
                                                                  log_k_bins=ST_lkb_growth)


ST_pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                     "ST_predicted_mass_100_scales_extended_low_mass_range.npy")
ST_mid_sig_sk, ST_f_nu_sk, ST_lkb_sk = get_f_nu_from_sk_prediction(ST_pred_sk, ic_small, rho_M, log_k_bins=ST_lkb_growth)

ST_k_predicted = np.load("/Users/lls/Documents/CODE/stored_files/hmf/upcrossings/small_box/"
                         "ST_k_predicted_barrier_sk_sigma.npy")
ST_k_predicted = np.load("/Users/lls/Documents/CODE/stored_files/hmf/upcrossings/small_box/"
                         "ST_k_predicted.npy")
ST_mig_sig_k, ST_f_nu_k, ST_lbk_k = get_f_nu_from_k_predictions(ST_k_predicted, ic_small, log_k_bins=ST_lkb)

# big box

pred_spherical_large_ST = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_ST_predicted_masses.npy")
ST_sig_large_sph, ST_f_nu_sph_large, ST_lbk_k_large = get_f_nu_from_spherical_prediction(pred_spherical_large_ST,
                                                                                         ic_200, rho_M, log_k_bins=30)

pred_sk_large_ST = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/volume_sharp_k/ALL_ST_predicted_masses"
                          ".npy")
m_sig_sk_large_ST, f_nu_sk_large_ST, ST_lbk_sk_large = get_f_nu_from_sk_prediction(pred_sk_large_ST, ic_200, rho_M,
                                                                                    log_k_bins=ST_lbk_k_large)


plt.plot(np.log(nu/delta_sc), np.log(f_ST), label="ST (theory)", color="k")
#plt.plot(ST_sig_sph_growth, np.log(ST_f_sph_growth), label="small")

plt.plot(ST_mid_sig_sk, np.log(ST_f_nu_sk), label="sk (small)")
# plt.plot(ST_mig_sig_k, np.log(ST_f_nu_k), label="k-pred (small)")

#plt.plot(ST_sig_large_sph, np.log(ST_f_nu_sph_large), label="spherical(large)")
plt.plot(m_sig_sk_large_ST, np.log(f_nu_sk_large_ST), label="sk (large)")

plt.legend(loc="best", frameon=False)
plt.xlabel(r"$\log [1/ \sigma ]$")
plt.ylabel(r"$\log [f(\nu)]$")
plt.ylim(-3, 0)
plt.xlim(-1.3, 1.2)
#plt.title("Press-Schechter")
#plt.tight_layout()


plt.plot(np.log(nu/delta_sc), np.log(f_ST), label="ST (theory)", color="k")
plt.plot(ST_sig_sph, np.log(ST_f_sph), label="spherical (small)")
plt.plot(ST_mid_sig_sk, np.log(ST_f_nu_sk), label="sk (small")
plt.plot(ST_mig_sig_k, np.log(ST_f_nu_k), label="k-pred (small)")

plt.plot(ST_sig_large_sph, np.log(ST_f_nu_sph_large), label="spherical(large)")
plt.plot(m_sig_sk_large_ST, np.log(f_nu_sk_large_ST), label="sk (large)")

plt.legend(loc="best", frameon=False)
plt.xlabel(r"$\log [1/ \sigma ]$")
plt.ylabel(r"$\log [f(\nu)]$")
plt.ylim(-3, 0)
plt.xlim(-1.3, 1.2)
#plt.title("Press-Schechter")
#plt.tight_layout()

fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twiny()
ax1.plot(np.log(nu/delta_sc), np.log(f_PS), label="PS", color="b", ls="--")
ax1.errorbar(sig_sph, np.log(f_sph), yerr=err_sph/f_sph, label=r" PS $D(a) = a$")

ax1.errorbar(np.log(nu/delta_sc), np.log(f_ST), label="ST", color="green", ls="--")
ax1.errorbar(ST_sig_sph, np.log(ST_f_sph), label=r"ST $D(a) = a$", color="green")

ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(xticklab(ax1.get_xticks()), fontsize=15)
ax2.set_xlabel(r"$\log(M)$")
ax1.legend(loc="best", frameon=False)
ax1.set_xlabel(r"$\log [1/ \sigma ]$")
ax1.set_ylabel(r"$\log [f(\nu)]$")
ax1.set_ylim(-5, 0)
fig.subplots_adjust(top=0.88)


fig, ax1 = plt.subplots(figsize=(9,6))
ax2 = ax1.twiny()
ax1.errorbar(np.log(nu/delta_sc), np.log(f_ST), label="ST", color="k", ls="--")
ax1.plot(ST_sig_sph, np.log(ST_f_sph), label=r" ST $D(a) = a$")
ax1.plot(ST_sig_sph_growth, np.log(ST_f_sph_growth), label=r"correct $D(a)$")
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(xticklab(ax1.get_xticks()), fontsize=15)
ax2.set_xlabel(r"$\log(M)$")
ax1.legend(loc="best", frameon=False)
ax1.set_xlabel(r"$\log [1/ \sigma ]$")
ax1.set_ylabel(r"$\log [f(\nu)]$")
ax1.set_ylim(-5, 0)
fig.subplots_adjust(top=0.88)
