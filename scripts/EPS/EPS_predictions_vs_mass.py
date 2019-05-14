import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from utils import mass
from utils import classification_results
from scripts.EPS import EPS_predictions as EPS_pred


if __name__ == "__main__":

    ids, predicted, true_labels = classification_results.load_classification_results()
    all_in_classified = ids[true_labels == 1]

    FNs_EPS = EPS_pred.get_subset_classified_particles_EPS(particles="false negatives")
    TPs_EPS = EPS_pred.get_subset_classified_particles_EPS(particles="true positives")

    FNs = classification_results.get_false_negatives(ids, predicted, true_labels)
    TPs = classification_results.get_true_positives(ids, predicted, true_labels)

    ### need to do two legends - one for false negatives/false positives and one for ML/EPS

    b, = mass.plot_fraction_subset_particles_per_halo_mass_bin(TPs_EPS, all_in_classified, label="TPs EPS",
                                                     xlabel=True, ylabel=True, number_of_bins=15,
                                                     xscale="equal total particles", color="b", marker='^', ls='--')

    d, = mass.plot_fraction_subset_particles_per_halo_mass_bin(FNs_EPS, all_in_classified, label="FNs EPS",
                                                          number_of_bins=15, xscale="equal total particles",
                                                          color='g', ls="--", marker="^")

    a, = mass.plot_fraction_subset_particles_per_halo_mass_bin(TPs, all_in_classified, label="TPs ML",
                                                               number_of_bins=15,
                                                          xscale="equal total particles", color='b')

    c, = mass.plot_fraction_subset_particles_per_halo_mass_bin(FNs, all_in_classified, label="FNs ML",
                                                          number_of_bins=15, xscale="equal total particles", color='g')

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

    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/mass_analysis"
                "/FNs_TPs_EPS_vs_ML_legend_try.pdf")