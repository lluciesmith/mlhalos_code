import sys

import scripts.hmf.hmf_theory

sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
from mlhalos import window
from scripts.hmf import halo_mass as hm
import matplotlib.pyplot as plt
from mlhalos import distinct_colours
from scripts.hmf import likelihood as lh


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
bins = np.arange(10, 15, 0.1)
m = 10**bins
mid_bins = (bins[1:] + bins[:-1])/2

tr_sk_vol = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sharp_k.npy")
w_sk = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250, volume="sharp-k")
m_sk_vol = mp.get_predicted_analytic_mass(w_sk.smoothing_masses, ic, barrier="spherical", trajectories=tr_sk_vol)
m_sk_vol_h = m_sk_vol * ic.initial_conditions.properties['h']
m_sk_vol_h.units = "Msol h^-1"
# np.save("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sharp_k_predicted_masses.npy",
# m_sk_vol_h)
hmf_sk = hm.get_empirical_number_halos(m_sk_vol_h, ic)
# Cannot rely hmf below M = 4.011 x 10^11 since below that r< 0.00557239 Mpc which is double the grid spacing.
hmf_sk_reliable = np.copy(hmf_sk)
hmf_sk_reliable[10**mid_bins <= 4.011 * 10**11] = 0


tr_sph_vol = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sphere.npy")
m_sph_vol = mp.get_predicted_analytic_mass(w_sk.smoothing_masses, ic, barrier="spherical", trajectories=tr_sph_vol)
m_sph_vol_h = m_sph_vol * ic.initial_conditions.properties['h']
m_sph_vol_h.units = "Msol h^-1"
hmf_sph = hm.get_empirical_number_halos(m_sph_vol_h, ic)

m_true, n_true = hm.get_true_number_halos_per_mass_bins(ic, bins)
m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel="PS")
poisson_n = [np.random.poisson(num_i, 10000) for num_i in num_th]
delta_m = np.diff(10**bins)


# plot for PS


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
colors = distinct_colours.get_distinct(4)

vplot = axes.violinplot(poisson_n, positions=m_th, widths=delta_m, showextrema=False, showmeans=False,
                       showmedians=False)
[b.set_color(colors[0]) for b in vplot['bodies']]
axes.step(m[:-1], num_th, where="post", color=colors[0], label="PS")
axes.plot([m[-2], m[-1]], [num_th[-1], num_th[-1]], color=colors[0])
plt.scatter(10**mid_bins, hmf_sk,  marker="^", color=colors[0], label="sharp-k volume", s=60, alpha=1)
plt.scatter(10**mid_bins, hmf_sk_orig,  marker="x", color=colors[0], label="sharp-k volume", s=60, alpha=1)
# plt.scatter(10**mid_bins, hmf_sk_reliable,  marker="^", color=colors[0], label="sharp-k volume", s=60, alpha=1)
# plt.scatter(10**mid_bins, hmf_sph,  marker="x", color=colors[0], label="sphere volume (250 radii)", s=30, alpha=1)
plt.scatter(10**mid_bins, n_true,  marker="o", color="k", s=10, alpha=1)
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Number of halos")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.xlim(5 * 10 ** 10, 10 ** 14)
plt.savefig("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/PS_250_radii_sk_vs_sphere_vol.png")

restr = (num_th >=10) & (m_th >= 8.23 * 10**10)
lik_sk = lh.chi_squared(hmf_sk_reliable[restr][hmf_sk_reliable[restr]!=0],
                       num_th[restr][hmf_sk_reliable[restr]!=0])
lik_sphere = lh.chi_squared(hmf_sph[restr][hmf_sph[restr]!=0], num_th[restr][hmf_sph[restr]!=0])

lik_theory_vs_true = lh.chi_squared(num_th[restr], n_true[restr])
lik_sk_vs_true = lh.chi_squared(hmf_sk_reliable[restr][hmf_sk_reliable[restr]!=0],
                               n_true[restr][hmf_sk_reliable[restr]!=0])
lik_sph_vs_true = lh.chi_squared(hmf_sph[restr][hmf_sph[restr]!=0],
                                n_true[restr][hmf_sph[restr]!=0])
