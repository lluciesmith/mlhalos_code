import numpy as np
from utils import plot
from mlhalos import distinct_colours
from mlhalos import parameters
import matplotlib.pyplot as plt


def get_median_properties(feature, log_mass_halos, bins, percentile=None):
    median_feature = np.zeros((len(bins)-1,))

    for i in range(len(bins) - 1):
        binning = (log_mass_halos >= bins[i]) & (log_mass_halos < bins[i + 1])
        f = feature[binning]

        if percentile is not None:
            q = np.percentile(feature, q=percentile)
            f = f[abs(f)< q]

        median_feature[i] = np.median(f)
    return median_feature


def get_std_properties(feature, log_mass_halos, bins, percentile=None):
    std_feature = np.zeros((len(bins)-1,))

    for i in range(len(bins) - 1):
        binning = (log_mass_halos >= bins[i]) & (log_mass_halos < bins[i + 1])
        f = feature[binning]

        if percentile is not None:
            q = np.percentile(feature, q=percentile)
            f = f[abs(f)< q]

        std_feature[i] = np.std(f)
    return std_feature

ell = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/ellipticity.npy")
den = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")
den_sub_ell = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/density_subtracted_ellipticity.npy")

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
den_particles = np.zeros((256**3, 50))
den_particles[ic.ids_IN, :] = den[:len(ic.ids_IN), :-1]
den_particles[ic.ids_OUT, :] = den[len(ic.ids_IN):, :-1]


halo_mass = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy")
ind = np.where(halo_mass>0)[0]

log_halos = np.log10(halo_mass[ind])
bin = np.log10(plot.get_log_spaced_bins_flat_distribution(halo_mass[ind], number_of_bins_init=25))

med_ell =get_median_properties(ell[ind, 25], log_halos, bin, percentile=98)
std_ell =get_std_properties(ell[ind, 25], log_halos, bin, percentile=98)
std_50_ell = get_std_properties(ell[ind, 25], log_halos, bin, percentile=50)

med_den =get_median_properties(den_particles[ind, 25], log_halos, bin)
std_den =get_std_properties(den_particles[ind, 25], log_halos, bin)
std_50_den = get_std_properties(den_particles[ind, 25], log_halos, bin, percentile=50)

med_den_sub_ell =get_median_properties(den_sub_ell[ind, 25], log_halos, bin)
std_den_sub_ell=get_std_properties(den_sub_ell[ind, 25], log_halos, bin)
std_50_den_sub_ell = get_std_properties(den_sub_ell[ind, 25], log_halos, bin, percentile=50)


h_400 = np.log10(ic.halo[400]['mass'].sum())

# plot

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 9), sharex=True)

colors = distinct_colours.get_distinct(4)

ax1.scatter((bin[:-1] + bin[1:])/2, med_ell, s=50, c=colors[0], label="ell")
ax1.fill_between((bin[:-1] + bin[1:]) / 2, med_ell - std_ell, med_ell + std_ell, alpha=0.1, color=colors[0])
ax1.fill_between((bin[:-1] + bin[1:]) / 2, med_ell - std_50_ell, med_ell + std_50_ell, alpha=0.4, color=colors[0])
ax1.legend(loc="best")

ax2.scatter((bin[:-1] + bin[1:])/2, med_den, s=50, c=colors[1], label="density")
ax2.fill_between((bin[:-1] + bin[1:]) / 2, med_den - std_50_den, med_den + std_50_den, alpha=0.1, color=colors[1])
ax2.fill_between((bin[:-1] + bin[1:]) / 2, med_den - std_50_den, med_den + std_50_den, alpha=0.4, color=colors[1])
ax2.legend(loc="best")

ax3.scatter((bin[:-1] + bin[1:])/2, med_den_sub_ell, s=50, c=colors[3], label="den-sub ell")
ax3.fill_between((bin[:-1] + bin[1:]) / 2, med_den_sub_ell - std_den_sub_ell, med_den_sub_ell + std_den_sub_ell,
                 alpha=0.1, color=colors[3])
ax3.fill_between((bin[:-1] + bin[1:]) / 2, med_den_sub_ell - std_50_den_sub_ell, med_den_sub_ell +
                 std_50_den_sub_ell, alpha=0.4, color=colors[3])
ax3.legend(loc="best")

ax1.axvline(x=h_400, ls="--", color="k")
ax2.axvline(x=h_400, ls="--", color="k")
ax3.axvline(x=h_400, ls="--", color="k")

fig.subplots_adjust(hspace=0)
ax2.set_ylabel(r"$\delta + 1$")
ax1.set_ylabel("e")
ax3.set_ylabel("den-sub e")
ax3.set_xlabel(r"$\log_{10} [\mathrm{M}/\mathrm{M_{\odot}}]$", labelpad=15)
ax1.set_ylim(-1,1)


plt.savefig("/Users/lls/Desktop/f_vs_mass_zoomed.png")