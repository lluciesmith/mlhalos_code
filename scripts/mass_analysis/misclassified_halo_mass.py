import sys
sys.path.append('/Users/lls/Documents/mlhalos_code')
from utils import classification_results
from utils.mass import plot_fraction_subset_particles_per_halo_mass_bin
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ids, predicted, true_labels = classification_results.load_classification_results()

    FNs = classification_results.get_false_negatives(ids, predicted, true_labels, threshold=None)
    TPs = classification_results.get_true_positives(ids, predicted, true_labels, threshold=None)
    all_in_classified = ids[true_labels == 1]

    plot_fraction_subset_particles_per_halo_mass_bin(TPs, FNs, all_in_classified, number_halos=True,
                                                     number_particles=True, xscale="equal total particles",
                                                     number_of_bins=25, label_TPs="TPs", label_FNs="FNs", marker='o',
                                                     ls='-', legend=True)

    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/mass_analysis"
                "/FNs_TPs_ratio_equal_total_particles_mass_bins_THRESHOLD_EPS.pdf")