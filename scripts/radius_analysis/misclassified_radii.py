"""

"""

import sys

sys.path.append('/Users/lls/Documents/mlhalos_code')
from utils.radius_func import plot_ratio_subset_vs_all_particles_per_r_bins
from utils import radius_func
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

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

    if sys.argv[1] == "bins of equal total particles":
        # plot the number of false negatives and the number of true positives as a function of log-spaced equal
        # number of total IN paticles radius bins

        plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, TPs_ratio, label="TPs", legend=True, xlabel=True,
                                                      ylabel=True, number_of_bins=15, xscale="equal total particles")
        plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, FNs_ratio, label="FNs", legend=True, number_of_bins=15,
                                                      xscale="equal total particles", color='g')
        # plt.savefig(
        #     "/Users/lls/Documents/CODE/stored_files/all_out/radii_files/plots/"
        #     "ratio_TPs_and_FNs_per_equal_particles_spaced_radius_bin.pdf")

    elif sys.argv[1] == "uniform bins":

        # plot the number of false negatives and the number of true positives as a function of log-spaced radius bins

        plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, TPs_ratio, label="TPs", legend=True, xlabel=True,
                                                      ylabel=True, number_of_bins=20, xscale="uniform",
                                                      legend_loc=(0.25, 0.05))
        plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, FNs_ratio, label="FNs", legend=True, number_of_bins=20,
                                                      xscale="uniform", legend_loc=(0.25, 0.05), color='g')

        # plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/plots/"
        #             "ratio_TPs_and_FNs_per_uniform_radius_bin.pdf")

    elif sys.argv[1] == "log-spaced bins":
        # plot the number of false negatives and the number of true positives as a function of uniformly spaced radius
        # bins

        plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, TPs_ratio, label="TPs", legend=True, xlabel=True,
                                                      ylabel=True, number_of_bins=20, xscale="log")
        plot_ratio_subset_vs_all_particles_per_r_bins(IN_ratio, FNs_ratio, label="FNs", legend=True, number_of_bins=20,
                                                      xscale="log", color='g')
        # plt.savefig(
        #     "/Users/lls/Documents/CODE/stored_files/all_out/radii_files/plots/ratio_TPs_and_FNs_per_log_radius_bin.pdf")
