import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code')
import utils.radius_func as rf
from utils import radius_func


for mass_bin in ["high", "mid", "small"]:
    FNs_radius_properties = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="false negatives",
                                                                                 mass_bin=mass_bin)
    TPs_radius_properties = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="true positives",
                                                                                 mass_bin=mass_bin)
    positives_radius_properties = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="positives",
                                                                                       mass_bin=mass_bin)

    FNs_EPS_radius_properties = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="false negatives",
                                                                                        mass_bin=mass_bin,
                                                                                        EPS_predictions=True)
    TPs_EPS_radius_properties = rf.extract_fraction_virial_radius_particles_in_mass_bin(particles="true positives",
                                                                                        mass_bin=mass_bin,
                                                                                        EPS_predictions=True)

    a, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(positives_radius_properties,
                                                                      TPs_radius_properties,
                                                                   number_of_bins=15, label="TPs ML",
                                                                   #legend=True,
                                                                   xscale="equal total particles", color='b',
                                                                   title=mass_bin.title() + "-mass bin"
                                                                   )
    c, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(positives_radius_properties,
                                                                      FNs_radius_properties,
                                                                   number_of_bins=15, xscale="equal total particles",
                                                                   label="FNs ML",
                                                                   #legend=True,
                                                                   xlabel=True, ylabel=True,
                                                                   color='g')
    b, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(positives_radius_properties,
                                                                   TPs_EPS_radius_properties,
                                                                   number_of_bins=15,
                                                                   xscale="equal total particles", marker='^',
                                                                   linestyle='--',
                                                                   color='b',
                                                                   label="TPs EPS"
                                                                    )
    d, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(positives_radius_properties,
                                                                   FNs_EPS_radius_properties,
                                                                   number_of_bins=15, xscale="equal total particles",
                                                                   color='g',
                                                                   marker='^', linestyle='--',
                                                                   label="FNs EPS")
    plt.ylim((0, 1))

    # legend

    categories = ["ML", "EPS"]

    e, = plt.plot([0], marker='None', linestyle='None', label='dummy-tophead')
    f, = plt.plot([0], marker='None', linestyle='None', label='dummy-empty')

    ML_marker_TP, = plt.plot([0], marker='o', color='b', linestyle='None', label='ML_marker')
    ML_marker_FN, = plt.plot([0], marker='o', color='g', linestyle='None', label='ML_marker')
    EPS_marker_TP, = plt.plot([0], marker='^', color='b', linestyle='None', label='EPS_marker')
    EPS_marker_FN, = plt.plot([0], marker='^', color='g', linestyle='None', label='EPS_marker')

    legend1 = plt.legend([e, (a, ML_marker_TP ), (b, EPS_marker_TP), e, (c, ML_marker_FN ), (d, EPS_marker_FN)],
                [r"\textbf{TPs}"] + categories + [r"\textbf{FNs}"] + categories,
                  loc="best", ncol=2, fontsize=12)

    # legend1 = plt.legend([e, f, e, f, (a, ML_marker_TP ), (b, EPS_marker_TP), (c, ML_marker_FN ), (d, EPS_marker_FN)],
    #               ['TPs', '', 'FNs', ''] + categories + categories,
    #               loc=4, ncol=4, columnspacing=0.2, frameon=False)  # Two columns, horizontal group labels

    plt.gca().add_artist(legend1)

    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/mass_analysis/FNs_TPs_" +
                mass_bin + "_mass_bin_vs_radius_ML_EPS_TRY.pdf", bbox_inches='tight')
    plt.clf()