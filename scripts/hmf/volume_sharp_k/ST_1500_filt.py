# import sys
# sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody

import scripts.hmf.hmf_theory
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
from mlhalos import window
from scripts.hmf import halo_mass as hm
import matplotlib.pyplot as plt
from mlhalos import distinct_colours
from scripts.hmf import likelihood as lh

######## HMF FOR 1500 FILTERING SCALES FOR SHARP K TRAJECTORIES WITH V = 6 pi^2 R^3 OR V=4/3 pi R^3


# Sheth-Tormen

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
bins = np.arange(10, 15, 0.1)
m_true, n_true = hm.get_true_number_halos_per_mass_bins(ic, bins)
m = 10**bins
mid_bins = (bins[1:] + bins[:-1])/2

ST_pred_mass_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                          "ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
hmf_sk = hm.get_empirical_number_halos(ST_pred_mass_sk, ic)

ST_pred_mass_sphere_volume = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/"
                                     "ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
hmf_sphere = hm.get_empirical_number_halos(ST_pred_mass_sphere_volume, ic)

m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel="ST")
poisson_n = [np.random.poisson(num_i, 10000) for num_i in num_th]

hmf_sk_ignore_first_bin = np.copy(hmf_sk)
hmf_sk_ignore_first_bin[11] = 0

# plot

colors = distinct_colours.get_distinct(4)
label="ST"
col = colors[3]
delta_m = np.diff(10**bins)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

vplot = axes.violinplot(poisson_n, positions=m_th, widths=delta_m, showextrema=False, showmeans=False,
                        showmedians=False)
[b.set_color(col) for b in vplot['bodies']]
axes.step(m[:-1], num_th, where="post", color=col, label=label)
axes.plot([m[-2], m[-1]], [num_th[-1], num_th[-1]], color=col)
plt.scatter(10 ** mid_bins, hmf_sk_ignore_first_bin, marker="^", color=col, label="sharp-k volume (1500 radii)", s=30,
            alpha=1)
plt.scatter(10**mid_bins, hmf_sphere,  marker="x", color=col, label="sphere volume (1500 radii)", s=30, alpha=1)
# plt.scatter(10**mid_bins, n_true,  marker="o", color="k", s=10, alpha=1)
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Number of halos")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.xlim(5 * 10 ** 10, 10 ** 14)
fig.get_tight_layout()
# fig.subplots_adjust(left=0.15)
plt.savefig("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/ST_1500_radii_sk_vs_sphere_vol.png")

# Likelihoods

restr = (num_th >=10) & (m_th >= 8.23 * 10**10)
lik_sk = lh.chi_squared(hmf_sk_ignore_first_bin[restr][hmf_sk_ignore_first_bin[restr]!=0],
                        num_th[restr][hmf_sk_ignore_first_bin[restr]!=0])
lik_sphere = lh.chi_squared(hmf_sphere[restr][hmf_sphere[restr]!=0], num_th[restr][hmf_sphere[restr]!=0])

lik_theory_vs_true = lh.chi_squared(num_th[restr], n_true[restr])
lik_sk_vs_true = lh.chi_squared(hmf_sk_ignore_first_bin[restr][hmf_sk_ignore_first_bin[restr]!=0],
                                n_true[restr][hmf_sk_ignore_first_bin[restr]!=0])
lik_sph_vs_true = lh.chi_squared(hmf_sphere[restr][hmf_sphere[restr]!=0],
                                 n_true[restr][hmf_sphere[restr]!=0])
