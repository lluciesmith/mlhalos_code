import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code')
import utils.radius_func as rf
from utils import radius_func


for mass_bin in ["high", "mid", "small"]:
    FNs_radius_fraction = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="false negatives",
                                                                                 mass_bin=mass_bin)
    TPs_radius_fraction = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="true positives",
                                                                                 mass_bin=mass_bin)
    positives_radius_fraction = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="positives",
                                                                                       mass_bin=mass_bin)

    FNs_EPS_radius_fraction = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="false negatives",
                                                                                        mass_bin=mass_bin,
                                                                                        EPS_predictions=True)
    TPs_EPS_radius_fraction = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="true positives",
                                                                                        mass_bin=mass_bin,
                                                                                        EPS_predictions=True)

    if mass_bin == "high":
        linestyle = '-'
    elif mass_bin == "mid":
        linestyle = '--'
    elif mass_bin == "small":
        linestyle = '-.'

    radius_func.plot_ratio_ML_EPS_per_r_bins(TPs_radius_fraction, TPs_EPS_radius_fraction, positives_radius_fraction,
                                             #label_scatter=["TPs" if mass_bin == "small" else None],
                                             xlabel=True,
                                             ylabel=r"$N_{\mathrm{ML}}/ N_{\mathrm{EPS}}$",
                                             number_of_bins=13, xscale="equal total particles", color="b",
                                             #legend=True,
                                             # title=mass_bin.title() + "-mass bin",
                                             linestyle=linestyle)

    radius_func.plot_ratio_ML_EPS_per_r_bins(FNs_radius_fraction, FNs_EPS_radius_fraction, positives_radius_fraction,
                                             #label_scatter=["FNs" if mass_bin == "small" else None], number_of_bins=13,
                                             xscale="equal total particles",
                                             color='g',
                                             #legend=True,
                                             ylabel=r"$N_{\mathrm{ML}}/ N_{\mathrm{EPS}}$",
                                             linestyle=linestyle)

# sort out legend

high_marker_TP, = plt.plot([0], color='k', linestyle='-', label='high_marker')
mid_marker_FN, = plt.plot([0], color='k', linestyle='--', label='mid_marker')
small_marker_TP, = plt.plot([0], color='k', linestyle='-.', label='small_marker')

FN_label, = plt.plot([0], color='g', marker='o', label='FN')
TP_label, = plt.plot([0], color='b', marker='o', label='TP')

legend1 = plt.legend([high_marker_TP, mid_marker_FN, small_marker_TP], ["high-mass", "mid-mass", "small-mass"],
                     loc="best")
legend2 = plt.legend([FN_label, TP_label], ["FNs", "TPs"])

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

plt.ylim((0.5, 1.4))

plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/mass_analysis"
                "/ratio_FNs_TPs_ML_EPS_radius_per_halo_mass_bin.pdf", bbox_inches='tight')

