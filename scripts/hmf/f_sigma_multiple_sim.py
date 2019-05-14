import sys
import numpy as np
import pynbody
from scripts.hmf import hmf_theory as ht
from scripts.hmf import hmf_simulation as hs
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
import matplotlib.pyplot as plt
from scripts.hmf import f_sig_PS as fsig
import scipy.interpolate


def violin_plot(xaxis, distributions_f_per_bisn, width_xbins, scatter_points, xbins, label_distr="orig",
                label_scatter="blind"):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    color="b"
    vplot = axes.violinplot(np.log(distributions_f_per_bisn), positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color) for b in vplot['bodies']]

    mean_distr = np.mean(distributions_f_per_bisn, axis=0)
    axes.step(xbins[:-1], np.log(mean_distr), where="post", color=color, label=label_distr)
    axes.plot([xbins[-2], xbins[-1]], [np.log(mean_distr[-1]), np.log(mean_distr[-1])], color=color)

    axes.scatter(xaxis, np.log(scatter_points), color="k", label=label_scatter)
    axes.set_ylim(-3,0)
    axes.set_xlim(-1.1, -0.1)

# blind sim

ic_blind = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/reseed50/IC.gadget3",
                                                  final_snapshot="/Users/lls/Documents/CODE/reseed50/snapshot_099",
                                                  load_final=True, path="/Users/lls/Documents/CODE/")
r_blind = np.load("/Users/lls/Documents/CODE/reseed50/PS_r_upcrossing.npy")


# orig sim

ic_orig = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
pred_spherical_growth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth"
                                "/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
rho_M = pynbody.analysis.cosmology.rho_M(ic_orig.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
r_orig = (pred_spherical_growth / (rho_M * 4 / 3 * np.pi)) ** (1 / 3)

# analysis

log_M_min = 10
log_M_max = 15
delta_log_M = 0.15

log_M_bins = np.arange(log_M_min, log_M_max, delta_log_M)
M_bins = (10 ** log_M_bins).view(pynbody.array.SimArray)
M_bins.units = "Msol h^-1"

v_bins = eb.calculate_variance(M_bins, ic_blind, cosmology="WMAP5", z=0)
sig_bins = np.sqrt(v_bins)
lnsig = np.log(1 / sig_bins)
dlnsig = np.diff(lnsig)
lnsig_mid = (lnsig[1:] + lnsig[:-1]) / 2

# theory

m_PS, num_halos_PS = ht.theoretical_number_halos(ic_orig, kernel="PS", cosmology="WMAP5", log_M_min=log_M_min,
                                                 log_M_max=log_M_max, delta_log_M=delta_log_M)
f_sig_PS, distr_PS = fsig.get_fsig_and_distr(m_PS, num_halos_PS, ic_orig, dlnsig)

# empirical

lnsig_mid_emp, f_empirical = fsig.get_f_empirical_from_r_predicted(r_orig, ic_orig, sig_bins)
lnsig_mid_emp, f_empirical_blind = fsig.get_f_empirical_from_r_predicted(r_blind, ic_blind, sig_bins)

f_empirical = f_empirical[::-1]
f_empirical_blind = f_empirical_blind[::-1]

f_orig_boot = fsig.get_f_sig_from_bootstrap_sub_volumes(r_orig, ic_orig, rho_M, sig_bins, dlnsig, num_bootstrap=1000,
                                                        num_sub=64)
f_blind_boot = fsig.get_f_sig_from_bootstrap_sub_volumes(r_blind, ic_blind, rho_M, sig_bins, dlnsig, num_bootstrap=1000)

mean_orig = np.mean(f_orig_boot, axis=0)
std_orig = np.std(f_orig_boot, axis=0)

mean_blind = np.mean(f_blind_boot, axis=0)
std_blind = np.std(f_blind_boot, axis=0)

m_min = ic_orig.initial_conditions['mass'].in_units("Msol h^-1")[0] * 100
restrict_PS_f = np.where((m_PS >= m_min) & (num_halos_PS >= 10))[0]
restrict_PS_bins = np.append(restrict_PS_f, restrict_PS_f[-1] + 1)


f_restrict_orig = f_orig_boot[:, restrict_PS_f]
f_restrict_blind = f_blind_boot[:, restrict_PS_f]

lnsig_restrict = lnsig_mid[restrict_PS_f]
lnsig_bins_restrict = lnsig[restrict_PS_bins]
bins_width_restricted = dlnsig[restrict_PS_f]

mean_orig_restricted = mean_orig[restrict_PS_f]
mean_blind_restricted = mean_blind[restrict_PS_f]

violin_plot(lnsig_restrict, f_restrict_orig, bins_width_restricted, mean_blind_restricted, lnsig_bins_restrict)
violin_plot(lnsig_restrict, f_restrict_blind, bins_width_restricted, mean_orig_restricted, lnsig_bins_restrict,
            label_distr="blind", label_scatter="orig")

C_blind = np.cov(f_restrict_blind.T)

f_restrict_orig = f_orig_boot[:, restrict_PS_f]
C_orig = np.cov(f_restrict_orig.T)

# C_blind = np.mat(np.cov(f_blind_boot.T, bias=True)[restrict_PS_f.min():restrict_PS_f.max() + 1,
#           restrict_PS_f.min():restrict_PS_f.max() +1])
# c_blind_coeff = np.corrcoef(f_blind_boot.T[restrict_PS_f])
#
# eigvals, eigvec = np.linalg.eig(c_blind_coeff)
# diagonal_C = np.diag(C_blind)
# y_d = np.mat(eigvec.T) /diagonal_C * np.mat(data_i)
#
#
# C_orig = np.mat(np.cov(f_orig_boot.T, bias=True)[restrict_PS_f.min():restrict_PS_f.max() + 1,
#          restrict_PS_f.min():restrict_PS_f.max() +1])
#
#
# # plot
#
def chi_squared(data, theory, covariance_inverse):
    # covariance = np.cov(data.T, bias=True)
    # c_inverse = np.linalg.inv(np.mat(covariance))
    chisq = np.zeros((len(data),))
    for i in range(len(data)):
        data_i = data[i]
        xd = np.mat(data_i - theory)
        chisq[i] = xd * np.mat(covariance_inverse) * xd.T
    return chisq
#
#
# def chi_squared_uncorrelated(data, theory, variance):
#     return ((data - theory)**2 /variance)
#
#
# chiq_sq_blind = chi_squared(f_blind_boot[:, restrict_PS_f], f_sig_PS[restrict_PS_f])
# chiq_sq_orig = chi_squared(f_orig_boot[:, restrict_PS_f], f_sig_PS[restrict_PS_f])
#
# plt.plot(lnsig_mid[restrict_PS_f], np.log(f_sig_PS[restrict_PS_f]), color="k", label="theory PS")
# plt.errorbar(lnsig_mid[restrict_PS_f], np.log(mean_orig[restrict_PS_f]),
#              yerr=std_orig[restrict_PS_f] / mean_orig[restrict_PS_f],
#              label="orig", color="b")
# plt.errorbar(lnsig_mid[restrict_PS_f], np.log(mean_blind[restrict_PS_f]),
#              yerr=std_blind[restrict_PS_f] / mean_blind[restrict_PS_f],
#              label="blind", color="g")
# # plt.scatter(lnsig_mid[restrict_PS_f], np.log(f_empirical[::-1][restrict_PS_f]), s=2)
# ylabel = r"$ f(\sigma) $"
# xlabel = r"$\ln \sigma^{-1}$"
# plt.ylabel(r"$ \ln $" + ylabel)
# plt.xlabel(xlabel)
# plt.legend(loc="lower left")
# plt.ylim(-3, 0)


