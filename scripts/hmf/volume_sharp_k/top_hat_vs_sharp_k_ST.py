import numpy as np

import scripts.hmf.hmf_theory
from scripts.hmf import predict_masses as mp
from mlhalos import parameters
from mlhalos import window
from scripts.hmf import halo_mass as hm
from mlhalos import distinct_colours
import matplotlib.pyplot as plt


def get_hmf_from_trajectories(initial_parameters, window_parameters, trajectories, barrier="ST"):
    pred_m= mp.get_predicted_analytic_mass(window_parameters.smoothing_masses, initial_parameters, barrier=barrier,
                                                  trajectories=trajectories)
    pred_m_h = pred_m * initial_parameters.initial_conditions.properties['h']
    pred_m_h.units = "Msol h^-1"
    hmf = hm.get_empirical_number_halos(pred_m_h, ic)
    return hmf


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)

bins = np.arange(10, 15, 0.1)
mid_bins = (bins[1:] + bins[:-1])/2
m = 10**bins
delta_m = np.diff(m)

# theory ST

m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel="ST")
poisson_n = [np.random.poisson(num_i, 10000) for num_i in num_th]

# TOP HAT 50 FILTERS

den_con = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/density_trajectories.npy")
hmf_th = get_hmf_from_trajectories(ic, w, den_con)
del den_con

# SHARP-K 50 FILTERS, SHARP K VOL

tr_50_sk_vol = np.load("/Users/lls/Documents/CODE/stored_files/hmf/traj_all_sharp_k_filter_shar_k_volume.npy")
hmf_sk_vol = get_hmf_from_trajectories(ic, w, tr_50_sk_vol)
del tr_50_sk_vol

# Cannot rely hmf below M = 4.011 x 10^11 Msol since below that r< 0.00557239 Mpc which is double the grid spacing.

w_sk = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50, volume="sharp-k")
m_sm = w.smoothing_masses * ic.initial_conditions.properties['h']
m_valid_min = m_sm[np.where(w_sk.smoothing_radii >= ic.boxsize_no_units *2 / ic.shape)[0][0]]

hmf_sk_reliable = np.copy(hmf_sk_vol)
hmf_sk_reliable[10**mid_bins < m_valid_min] = 0

# SHARP-K 50 FILTERS, TOP_HAT VOLUME

tr_50_sph_vol = np.load("/Users/lls/Documents/CODE/stored_files/hmf/traj_all_sharp_k_filter_volume_sphere.npy")
hmf_sph_vol = get_hmf_from_trajectories(ic, w, tr_50_sph_vol)
del tr_50_sph_vol

# plot

colors = distinct_colours.get_distinct(4)
label="ST"
col = colors[3]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

vplot = axes.violinplot(poisson_n, positions=m_th, widths=delta_m, showextrema=False, showmeans=False,
                        showmedians=False)
[b.set_color(col) for b in vplot['bodies']]
axes.step(m[:-1], num_th, where="post", color=col, label=label)
axes.plot([m[-2], m[-1]], [num_th[-1], num_th[-1]], color=col)
plt.scatter(10**mid_bins, hmf_th,  marker="o", color=col, label="top hat", s=20, alpha=1)
plt.scatter(10 ** mid_bins, hmf_sk_reliable, marker="^", color=col, label="sharp-k (sharp-k vol)", s=20,
            alpha=1)
plt.scatter(10**mid_bins, hmf_sph_vol,  marker="x", color=col, label="sharp k (vol sphere)", s=20, alpha=1)
# plt.scatter(10**mid_bins, n_true,  marker="o", color="k", s=10, alpha=1)
plt.legend(loc="best")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Number of halos")
plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
plt.xlim(5 * 10 ** 10, 10 ** 14)
fig.get_tight_layout()
plt.savefig("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/top_hat_vs_sharp_k_ST.png")