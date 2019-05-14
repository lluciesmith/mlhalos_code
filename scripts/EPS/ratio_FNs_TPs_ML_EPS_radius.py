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

    # GET EQUIVALENT RATIO FOR RADIUS

    radius_func.plot_ratio_ML_EPS_per_r_bins(TPs_ratio, TPs_EPS_ratio, IN_ratio,
                                                  label="TPs",
                                                  xlabel=True,
                                                  ylabel=r"$N_{\mathrm{ML}}/ N_{\mathrm{EPS}}$",
                                                  #ylabel=r"$\mathrm{N_{ML} / N_{EPS}}$",
                                                  number_of_bins=13,
                                                  xscale="equal total particles", color="b",
                                                  legend=True)

    radius_func.plot_ratio_ML_EPS_per_r_bins(FNs_ratio, FNs_EPS_ratio, IN_ratio, label="FNs",
                                                   number_of_bins=13, xscale="equal total particles",
                                                   color='g', legend=True,
                                                  ylabel=r"$N_{\mathrm{ML}}/ N_{\mathrm{EPS}}$"
                                                  #ylabel=r"$\mathrm{N_{ML} / N_{EPS}}$"
                                                  )

    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/mass_analysis"
                "/ratio_FNs_TPs_ML_EPS_radius_2.pdf")