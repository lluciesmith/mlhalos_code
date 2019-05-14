import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from utils import mass
from utils import classification_results
from scripts.EPS import EPS_predictions as EPS_pred
import importlib
importlib.reload(EPS_pred)
from mlhalos import distinct_colours

if __name__ == "__main__":

    ids, predicted, true_labels = classification_results.load_classification_results()
    all_in_classified = ids[true_labels == 1]

    FNs_EPS = EPS_pred.get_subset_classified_particles_EPS(particles="false negatives")
    TPs_EPS = EPS_pred.get_subset_classified_particles_EPS(particles="true positives")

    FNs = classification_results.get_false_negatives(ids, predicted, true_labels)
    TPs = classification_results.get_true_positives(ids, predicted, true_labels)

    ### need to do two legends - one for false negatives/false positives and one for ML/EPS

    # mass.plot_ratio_ML_EPS_subset_particles_per_halo_mass_bin(TPs, TPs_EPS, all_in_classified, label="TPs",
    #                                                           xlabel=True,
    #                                                           ylabel=r"$N_{\mathrm{ML}}/ N_{\mathrm{EPS}}$",
    #                                                           number_of_bins=15, xscale="equal total particles",
    #                                                           color="b", legend=True)
    #
    # mass.plot_ratio_ML_EPS_subset_particles_per_halo_mass_bin(FNs, FNs_EPS, all_in_classified, label="FNs",
    #                                                           number_of_bins=15, xscale="equal total particles",
    #                                                           color='g', legend=True,
    #                                                           ylabel=r"$N_{\mathrm{ML}}/ N_{\mathrm{EPS}}$")

    mass.plot_TPS_FPs_particles_per_halo_mass_bin(all_in_classified, TPs, FNs, legend=False)
    mass.plot_TPS_FPs_particles_per_halo_mass_bin(all_in_classified, TPs_EPS, FNs_EPS, label_TPs="TPs EPS",
                                                  label_FNs="FNs EPS", marker='^', ls='--', legend=False)

    # legend

    categories = ["ML", "EPS"]

    e, = plt.plot([0], marker='None', linestyle='None', label='dummy-tophead')
    f, = plt.plot([0], marker='None', linestyle='None', label='dummy-empty')

    colors = distinct_colours.get_distinct(2)
    ML_marker_TP, = plt.plot([0], marker='o', color=colors[1], linestyle='-', label='ML_marker')
    ML_marker_FN, = plt.plot([0], marker='o', color=colors[0], linestyle='-', label='ML_marker')
    EPS_marker_TP, = plt.plot([0], marker='^', color=colors[1], linestyle='--', label='EPS_marker')
    EPS_marker_FN, = plt.plot([0], marker='^', color=colors[0], linestyle='--', label='EPS_marker')

    legend1 = plt.legend([e, ML_marker_TP, EPS_marker_TP, e, ML_marker_FN, EPS_marker_FN],
                         [r"\textbf{TPs}"] + categories + [r"\textbf{FNs}"] + categories,
                         loc="best", ncol=2, fontsize=12)

    # legend1 = plt.legend([e, f, e, f, (a, ML_marker_TP ), (b, EPS_marker_TP), (c, ML_marker_FN ), (d, EPS_marker_FN)],
    #               ['TPs', '', 'FNs', ''] + categories + categories,
    #               loc=4, ncol=4, columnspacing=0.2, frameon=False)  # Two columns, horizontal group labels

    plt.gca().add_artist(legend1)

    # plt.savefig("/Users/lls/Desktop/ratio_FNs_TPs_ML_EPS_mass_correct_binning_2.pdf")

    # e, = plt.plot([0], color = c[1], label="trajectories+EPS label")