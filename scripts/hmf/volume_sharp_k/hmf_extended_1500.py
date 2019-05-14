import scripts.hmf.hmf_theory
from mlhalos import distinct_colours
from mlhalos import parameters
import numpy as np
import matplotlib.pyplot as plt
from scripts.hmf import halo_mass as hm
from scripts.hmf import likelihood as lh


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
delta_log_M = 0.15
bins = np.arange(10, 15, delta_log_M)
m = 10**bins
mid_bins = (bins[1:] + bins[:-1])/2

m_true, n_true = hm.get_true_number_halos_per_mass_bins(ic, bins)

# Press-Schechter

m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel="PS", delta_log_M=delta_log_M)
poisson_n = [np.random.poisson(num_i, 10000) for num_i in num_th]
delta_m = np.diff(10**bins)

PS_pred_mass_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                          "PS_predicted_mass_100_scales_extended_low_mass_range.npy")
hmf_sk = hm.get_empirical_number_halos(PS_pred_mass_sk, ic, delta_log_M=delta_log_M)

hmf_sk_ignore_first_bin = np.copy(hmf_sk)
hmf_sk_ignore_first_bin[5] = 0

colors = distinct_colours.get_distinct(4)
label="PS"
col = colors[0]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

vplot = axes.violinplot(poisson_n, positions=m_th, widths=delta_m, showextrema=False, showmeans=False,
                        showmedians=False)
[b.set_color(col) for b in vplot['bodies']]
axes.step(m[:-1], num_th, where="post", color=col, label=label)
axes.plot([m[-2], m[-1]], [num_th[-1], num_th[-1]], color=col)
# plt.scatter(10 ** mid_bins, hmf_sk_ignore_first_bin, marker="^", color=col, label="empirical", s=30,
#             alpha=1)
plt.scatter(10 ** mid_bins, hmf_sk, marker="^", color=col, label="empirical", s=30,
            alpha=1)
plt.scatter(10**mid_bins, n_true,  marker="o", color="k", s=10, alpha=1, label="sim")
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Number of halos")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.xlim(6.3 * 10 ** 10, 10 ** 14)

# likelihood

restr = (num_th >=10) & (m_th >= 5.77 * 10**10)
lik_sk = lh.chi_squared(hmf_sk_ignore_first_bin[restr][hmf_sk_ignore_first_bin[restr] != 0],
                        num_th[restr][hmf_sk_ignore_first_bin[restr] != 0])

# Sheth-Tormen

ST_pred_mass_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                          "ST_predicted_mass_100_scales_extended_low_mass_range.npy")
hmf_sk = hm.get_empirical_number_halos(ST_pred_mass_sk, ic)

m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel="ST")
poisson_n = [np.random.poisson(num_i, 10000) for num_i in num_th]

hmf_sk_ignore_first_bin = np.copy(hmf_sk)
hmf_sk_ignore_first_bin[7] = 0

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
# plt.scatter(10**mid_bins, n_true,  marker="o", color="k", s=10, alpha=1)
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Number of halos")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.xlim(5 * 10 ** 10, 10 ** 14)
fig.get_tight_layout()

restr = (num_th >=10) & (m_th >= 5.77 * 10**10)
lik_sk = lh.chi_squared(hmf_sk_ignore_first_bin[restr][hmf_sk_ignore_first_bin[restr] != 0],
                        num_th[restr][hmf_sk_ignore_first_bin[restr] != 0])