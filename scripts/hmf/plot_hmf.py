import numpy as np
import matplotlib.pyplot as plt
from mlhalos import parameters
from mlhalos import window
from scripts.hmf import hmf



def plot_differential_ratio_simulation_vs_theory_and_empirical(mid_mass, simulation_hmf, theory_hmf, empirical_hmf,
                                                               type="EPS"):
    if type == "EPS":
        labels = ["EPS theory", "EPS empirical"]
        plt.ylabel(r"$(n_{EPS} - n_{sim})/n_{sim}$")
    elif type == "ST":
        labels = ["ST theory", "ST empirical"]
        plt.ylabel(r"$(n_{ST} - n_{sim})/n_{sim}$")

    # diff_ratio_sim_theory = (simulation_hmf[simulation_hmf != 0] - theory_hmf[simulation_hmf != 0]) / theory_hmf[
    #     simulation_hmf != 0]
    # diff_ratio_sim_emp = (simulation_hmf[simulation_hmf != 0] - empirical_hmf[simulation_hmf != 0]) / empirical_hmf[
    #     simulation_hmf != 0]
    diff_ratio_sim_theory = (simulation_hmf[simulation_hmf != 0] - theory_hmf[simulation_hmf != 0]) / theory_hmf[
        simulation_hmf != 0]
    diff_ratio_sim_emp = (simulation_hmf[simulation_hmf != 0] - empirical_hmf[simulation_hmf != 0]) / empirical_hmf[
        simulation_hmf != 0]

    plt.scatter(10 ** mid_mass[simulation_hmf != 0][:-2], diff_ratio_sim_theory[:-2], color="g"
                # ,label="theory/sim"
                )
    plt.plot(10 ** mid_mass[simulation_hmf != 0][:-2], diff_ratio_sim_theory[:-2], color="g", label=labels[0]
             )
    plt.scatter(10 ** mid_mass[simulation_hmf != 0][:-2], diff_ratio_sim_emp[:-2], color="b"
                # ,label="theory/sim"
                )
    plt.plot(10 ** mid_mass[simulation_hmf != 0][:-2], diff_ratio_sim_emp[:-2], color="b", label=labels[1]
             )
    plt.axhline(y=0, color="k")
    # plt.plot(10 ** mid_mass[:-1], ST_hmf[:-1] / ST_dn_dlog10m[:-1], color="b", label="theory/empirical")
    plt.xscale("log")
    plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
    plt.legend(loc="best")





# trajectories = np.load("/Users/lls/Documents/CODE/stored_files/hmf/traj_all_sharp_k_filter_shar_k_volume.npy")
trajectories = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sharp_k.npy")

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250)

# Returns log-scale window smoothing masses in units [Msol h^-1]
log_mass, mid_mass = hmf.get_log_bin_masses_and_mid_values(initial_parameters=ic, window_parameters=w)
delta_log_mass = log_mass[2] - log_mass[1]

# predicted halo mass functions from mass labels

# traj_250 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_250_bins.npy")
# n_density_halos_ST = get_predicted_number_density_halos_in_mass_bins(log_mass, mid_mass, kernel="ST",
#                                                                      initial_parameters=ic, trajectories=trajectories)
n_density_halos_PS = get_predicted_number_density_halos_in_mass_bins(log_mass, mid_mass, kernel="PS",
                                                                     initial_parameters=ic, trajectories=trajectories)
#
# ST_dn_dlog10m = get_dn_dlog10m(n_density_halos_ST, 10**log_mass, 10**mid_mass)
PS_dn_dlog10m = get_dn_dlog10m(n_density_halos_PS, 10 ** log_mass, 10 ** mid_mass)

# Theoretical halo mass functions using ST and EPS kernels

# M, sigma, ST_hmf = hmf.hmf_theory(ic, z=0, cosmology="WMAP5", kernel="ST", log_M_min=log_mass.min(),
#                               log_M_max=log_mass.max() + delta_log_mass, delta_log_M=delta_log_mass)
M, sigma, PS_hmf = hmf_theory(ic, z=0, cosmology="WMAP5", kernel="PS", filter="sharp k",
                              log_M_min=log_mass.min(),
                              log_M_max=log_mass.max(), delta_log_M=delta_log_mass)

# "True" calculated halo mass function

log_m_true, log_M_mid_true, n_true_density_halos = ground_truth_number_of_halos_per_mass_bin(ic, w,
                                                                                             masses_type="smoothing")
delta_log_M_true = log_M_mid_true[2] - log_M_mid_true[1]
sim_dn_dlog10m = get_dn_dlog10m(n_true_density_halos, 10 ** log_m_true, 10 ** log_M_mid_true)

# We want to plot the comoving number density of halos (per unit log mass)

# plt.loglog(10 ** mid_mass, ST_dn_dlog10m, color="b", label="ST labels")
# plt.loglog(10 ** mid_mass, ST_hmf, color="b", ls="--", label="ST theoretical")

plt.loglog(10 ** mid_mass, PS_hmf, color="b", label="PS theory")
plt.loglog(10 ** mid_mass, sim_dn_dlog10m, color="k", label="sim")
plt.loglog(10 ** mid_mass, PS_dn_dlog10m, color="g", label="EPS labels")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.ylabel(r"$\mathrm{\mathrm{dN} / \mathrm{d} \log_{10} \mathrm{M}} [\mathrm{Mpc}^{-3} \mathrm{h}^3 \mathrm{a}^{-3}]$")
plt.legend(loc="best")
# plt.savefig("/Users/lls/Documents/CODE/stored_files/hmf/PS_sharp_k_vol_sharp_k.png")
#

# ratio
# fig = plt.figure(figsize=(10,8))
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)
#
# # gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
# #
# # ax1 = plt.subplot(gs[0])
# # ax2 = plt.subplot(gs[1])
#
# ax2.plot(10 ** mid_mass[:-2], ST_hmf[:-2]/ST_dn_dlog10m[:-2], color="b", label="ST")
# ax2.plot(10 ** mid_mass[:-2], PS_hmf[:-2]/PS_dn_dlog10m[:-2], color="g", label="EPS")
# ax2.set_xscale("log")
# #ax2.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
# ax2.set_ylabel("theory/empirical")
# ax2.legend(loc="best")
# #ax2.set_xlim(10**10, 10**14)
# #ax2.set_ylim(0, 6)
#
# ax1.plt.loglog(10 ** mid_mass, ST_dn_dlog10m, color="b", label="ST labels")
# ax1plt.loglog(10 ** mid_mass, PS_dn_dlog10m, color="g", label="EPS labels")
# # plt.scatter(10**(mid_mass), ST_dn_dlog10m, color="b", label="ST labels")
# # plt.scatter(10**(mid_mass), PS_dn_dlog10m, color="g", label="EPS labels")
# fig.subplots_adjust(hspace=0)
# ax1.loglog(M[:-2], ST_hmf[:-2], color="b", ls="--", label="ST theoretical")
# ax1.loglog(M[:-2], PS_hmf[:-2], color="g", ls="--", label="EPS theoretical")
# # ax1.loglog(M, ST_hmf*0.322/np.sqrt(0.707), color="b", ls="--", label="ST theoretical")
# # ax1.loglog(M, PS_hmf/0.5, color="g", ls="--", label="EPS theoretical")
#
# #plt.loglog(10 ** log_M_mid_true, sim_dn_dlog10m, color="k",  label="sim")
# #ax1.set_ylim(10**-5, 10**0)
#
# ax2.set_xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
# ax1.set_ylabel(r"$\mathrm{\mathrm{dN} / \mathrm{d} \log_{10} \mathrm{M}} [\mathrm{Mpc}^{-3} \mathrm{h}^3 \mathrm{"
#             r"a}^{-3}]$")
# ax1.legend(loc="best")

# plt.clf()
# #plt.plot(10 ** mid_mass[:-1], ST_hmf[:-1] / sim_dn_dlog10m[:-1], color="b", label="ST")
# plt.plot(10 ** mid_mass, PS_hmf / sim_dn_dlog10m, color="g", label="theory/sim")
# plt.plot(10 ** mid_mass, PS_hmf / PS_dn_dlog10m, color="b", label="theory/empirical")
# plt.xscale("log")
# plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
# #plt.ylabel("theory/empirical")
# plt.legend(loc="best")
# plt.title("EPS")
# plt.tight_layout()
# #plt.savefig("/Users/lls/Desktop/th_vs_emp_EPS.png")
# #plt.savefig("/Users/lls/Desktop/hmfs_all.png")
#
# plot_differential_ratio_simulation_vs_theory_and_empirical(mid_mass, sim_dn_dlog10m, ST_hmf, ST_dn_dlog10m,
#                                                            type="ST")
# plot_differential_ratio_simulation_vs_theory_and_empirical(mid_mass, sim_dn_dlog10m, PS_hmf, PS_dn_dlog10m,type="EPS")
