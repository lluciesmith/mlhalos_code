import numpy as np
import sys

import scripts.hmf.hmf_theory

sys.path.append("/Users/lls/Documents/mlhalos_code")
from mlhalos import parameters
from scripts.hmf import halo_mass as hm
import matplotlib.pyplot as plt
from mlhalos import distinct_colours


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
kernel = "ST"
delta_log_M = 0.1

bins = np.arange(10, 15, delta_log_M)
m = 10**bins
mid_bins = (bins[1:] + bins[:-1])/2

if kernel == "PS":
    m_sk_vol_h = np.load("/Users/lls/Documents/CODE/stored_files/hmf/"
                         "250_traj_all_sharp_k_filter_volume_sharp_k_predicted_masses.npy")
    pred_low_mass = np.load("/Users/lls/Documents/CODE/stored_files/hmf/PS_all_predictions_in_low_mass_range.npy")
    num_color = 0

else:
    m_sk_vol_h = np.load("/Users/lls/Documents/CODE/stored_files/hmf"
                         "/250_ST_traj_all_sharp_k_filter_volume_sharp_k_predicted_masses.npy")
    pred_low_mass = np.load("/Users/lls/Documents/CODE/stored_files/hmf/ST_all_predictions_in_low_mass_range.npy")
    num_color = 3

ind = np.logical_or(np.isnan(m_sk_vol_h), (m_sk_vol_h<=1.4 * 10**11))

m_pred = np.copy(m_sk_vol_h)
m_pred[ind] = pred_low_mass[ind]

hmf_sk = hm.get_empirical_number_halos(m_pred, ic, delta_log_M=delta_log_M)

hmf_sk_ignore_first_bin = np.copy(hmf_sk)
hmf_sk_ignore_first_bin[7] = 0

m_true, n_true = hm.get_true_number_halos_per_mass_bins(ic, bins)
m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel=kernel, delta_log_M=delta_log_M)
poisson_n = [np.random.poisson(num_i, 10000) for num_i in num_th]
delta_m = np.diff(10**bins)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
colors = distinct_colours.get_distinct(4)

vplot = axes.violinplot(poisson_n, positions=m_th, widths=delta_m, showextrema=False, showmeans=False,
                       showmedians=False)
[b.set_color(colors[num_color]) for b in vplot['bodies']]
axes.step(m[:-1], num_th, where="post", color=colors[num_color], label="PS")
axes.plot([m[-2], m[-1]], [num_th[-1], num_th[-1]], color=colors[num_color])
plt.scatter(10**mid_bins, hmf_sk,  marker="^", color=colors[num_color], label="sharp-k volume", s=60, alpha=1)
# plt.scatter(10**mid_bins, hmf_sk_reliable,  marker="^", color=colors[0], label="sharp-k volume", s=60, alpha=1)
# plt.scatter(10**mid_bins, hmf_sph,  marker="x", color=colors[0], label="sphere volume (250 radii)", s=30, alpha=1)
plt.scatter(10**mid_bins, n_true,  marker="o", color="k", s=10, alpha=1)
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Number of halos")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.xlim(5 * 10 ** 10, 10 ** 14)

restr = (num_th >=10) & (m_th >= 5.77 * 10**10)
restr = (num_th >=10) & (m_th >= 4.11 * 10**11)

lik_sk = lh.chi_squared(hmf_sk_ignore_first_bin[restr][hmf_sk_ignore_first_bin[restr] != 0],
                        num_th[restr][hmf_sk_ignore_first_bin[restr] != 0])

