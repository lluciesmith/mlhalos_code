import sys

import matplotlib.pyplot as plt

import utils.radius_func
from utils.radius_func import extract_fraction_virial_radius_particles_in_mass_bin

sys.path.append('/Users/lls/Documents/mlhalos_code')

if __name__ == "__main__":

    for mass_bin in ["high", "mid", "small"]:
        FNs_radius_properties = extract_fraction_virial_radius_particles_in_mass_bin(particles="false negatives",
                                                                                     mass_bin=mass_bin)
        TPs_radius_properties = extract_fraction_virial_radius_particles_in_mass_bin(particles="true positives",
                                                                                     mass_bin=mass_bin)
        positives_radius_properties = extract_fraction_virial_radius_particles_in_mass_bin(particles="positives",
                                                                                           mass_bin=mass_bin)

        utils.radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(positives_radius_properties, TPs_radius_properties,
                                                                        number_of_bins=15, label="TPs", legend=True,
                                                                        xscale="equal total particles",
                                                                        title=mass_bin.title() + "-mass bin")
        utils.radius_func.plot_ratio_subset_vs_all_particles_per_r_bins(positives_radius_properties, FNs_radius_properties,
                                                                        number_of_bins=15, xscale="equal total particles",
                                                                        label="FNs", legend=True, xlabel=True, ylabel=True, color='g')

        plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/mass_analysis/FNs_TPs_" +
                    mass_bin + "_mass_bin_vs_15_radius_bins.pdf", bbox_inches='tight')
        plt.clf()
