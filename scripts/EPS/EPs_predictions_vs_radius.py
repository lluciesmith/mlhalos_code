import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from utils import radius_func
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # ML - ALGO

    FNs_radii_properties = radius_func.extract_radii_properties_subset_particles(particles="false negatives")
    TPs_radii_properties = radius_func.extract_radii_properties_subset_particles(particles="true positives")

    IN_radii_properties = radius_func.load_radii_properties(particles_class="in", feature_set="test")

    # ignore ratio radius/virial radius of halo 336 & exclude all particles with r < softening-length, i.e. r < 25.6 kpc

    FNs_ratio = FNs_radii_properties[:, 2][np.logical_and(FNs_radii_properties[:, 2] != 0,
                                                          FNs_radii_properties[:, 1] > 25.6)]
    TPs_ratio = TPs_radii_properties[:, 2][np.logical_and(TPs_radii_properties[:, 2] != 0,
                                                          TPs_radii_properties[:, 1] > 25.6)]
    IN_ratio = IN_radii_properties[:, 2][np.logical_and(IN_radii_properties[:, 2] != 0,
                                                        IN_radii_properties[:, 1] > 25.6)]

    # EPS

    FNs_EPS_radii_properties = radius_func.extract_radii_properties_subset_particles(particles="false negatives",
                                                                                     EPS_predictions=True)
    TPs_EPS_radii_properties = radius_func.extract_radii_properties_subset_particles(particles="true positives",
                                                                                     EPS_predictions=True)

    FNs_EPS_ratio = FNs_EPS_radii_properties[:, 2][np.logical_and(FNs_EPS_radii_properties[:, 2] != 0,
                                                              FNs_EPS_radii_properties[:, 1] > 25.6)]
    TPs_EPS_ratio = TPs_EPS_radii_properties[:, 2][np.logical_and(TPs_EPS_radii_properties[:, 2] != 0,
                                                              TPs_EPS_radii_properties[:, 1] > 25.6)]

    # plot the number of false negatives and the number of true positives as a function of log-spaced equal
    # number of total IN paticles radius bins

    a, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, TPs_ratio, label="TPs ML",
                                                              xlabel=True, ylabel=True, number_of_bins=15,
                                                              xscale="equal total particles")

    c, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, FNs_ratio, label="FNs ML",
                                                              number_of_bins=15, xscale="equal total particles",
                                                              color='g')

    b, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, TPs_EPS_ratio, number_of_bins=15,
                                                              xscale="equal total particles", color='b', marker="^",
                                                              linestyle="--", label="TPs EPS")
    d, = radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, FNs_EPS_ratio,
                                                              number_of_bins=15, xscale="equal total particles",
                                                              color='g', marker="^", linestyle="--", label="FNs EPS")

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

    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/plots/ratio_TPs_and_FNs_EPS_vs_ML.pdf")