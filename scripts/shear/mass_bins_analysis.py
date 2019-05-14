import numpy as np
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
from utils import mass
from mlhalos import machinelearning as ml
from mlhalos import plot
import matplotlib.pyplot as plt
from mlhalos import distinct_colours
import collections


def plot_rocs(fpr, tpr, ls="-", cols="b"):
    l, = plt.plot(fpr, tpr, lw=1.5, ls=ls, color=cols)
    return l,


def get_roc_mass_bins(ids, predicted_label, true_label):
    FPR = np.zeros((50, 3))
    TPR = np.zeros((50, 3))
    AUC = []

    mass_bins = ["high", "mid", "small"]
    for j in range(3):
        mass_bin = mass_bins[j]
        index = mass.get_indices_of_particles_array_in_mass_bin(ids, mass_bin=mass_bin)
        predicted_mass_bin = predicted_label[index, :]
        true_label_mass_bin = true_label[index]

        fpr_mass_bin, tpr_mass_bin, auc_mass_bin, th = ml.roc(predicted_mass_bin, true_label_mass_bin)

        FPR[:, j] = fpr_mass_bin
        TPR[:, j] = tpr_mass_bin
        AUC.append(auc_mass_bin)

    return FPR, TPR, AUC


def get_legend_for_mass_bins(a, b, c, d, g, h, cols):
    categories = ["density+shear", "density"]

    e, = plt.plot([0], marker='None', linestyle='None', label='dummy-tophead')
    f, = plt.plot([0], marker='None', linestyle='None', label='dummy-empty')

    den_marker_high, = plt.plot([0], marker=None, color=cols[0], linestyle='--', label='density_marker')
    den_marker_mid, = plt.plot([0], marker=None, color=cols[1], linestyle='--', label='density_marker')
    den_marker_small, = plt.plot([0], marker=None, color=cols[2], linestyle='--', label='density_marker')
    shear_marker_high, = plt.plot([0], marker=None, color=cols[0], linestyle='-', label='shear_marker')
    shear_marker_mid, = plt.plot([0], marker=None, color=cols[1], linestyle='-', label='shear_marker')
    shear_marker_small, = plt.plot([0], marker=None, color=cols[2], linestyle='-', label='shear_marker')

    legend1 = plt.legend([e, (a, shear_marker_high), (c, shear_marker_mid), (g, shear_marker_small),
                          e, (b, den_marker_high), (d, den_marker_mid), (h, den_marker_small)],
                         [categories[0]] + ["high" + ' (%.3f)' % (AUC_both[0][0]), "mid" + ' (%.3f)' % (AUC_both[0][1]),
                                            "small" + ' (%.3f)' % (AUC_both[0][2])] + [categories[1]]
                         + ["high" + ' (%.3f)' % (AUC_both[1][0]), "mid" + ' (%.3f)' % (AUC_both[1][1]),
                            "small" + ' (%.3f)' % (AUC_both[1][2])],
                         loc="best", ncol=2, fontsize=18)
    return legend1


def plot_rocs_with_appropriate_legend_for_mass_bins(FPR_both, TPR_both, AUC_both):
    figure, ax = plt.subplots(figsize=(9, 7))
    cols = distinct_colours.get_distinct(3)

    a, = plot_rocs(FPR_both[:, 0], TPR_both[:, 0], ls="-", cols=cols[0])
    b, = plot_rocs(FPR_both[:, 3], TPR_both[:, 3], ls="--", cols=cols[0])
    c, = plot_rocs(FPR_both[:, 1], TPR_both[:, 1], ls="-", cols=cols[1])
    d, = plot_rocs(FPR_both[:, 4], TPR_both[:, 4], ls="--", cols=cols[1])
    g, = plot_rocs(FPR_both[:, 2], TPR_both[:, 2], ls="-", cols=cols[2])
    h, = plot_rocs(FPR_both[:, 5], TPR_both[:, 5], ls="--", cols=cols[2])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    legend1 = get_legend_for_mass_bins(a, b, c, d, g, h, cols)
    plt.gca().add_artist(legend1)

if __name__ == "__main__":

    # y_predicted_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/predicted_den+den_sub_ell.npy")
    # y_true_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/true_den+den_sub_ell.npy")
    y_predicted_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol"
                                "/predicted_den+den_sub_ell+den_sub_prol.npy")
    y_true_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol"
                           "/true_den+den_sub_ell+den_sub_prol.npy")

    y_predicted_den = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/predicted_den"
                              ".npy")
    y_true_den = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/true_den.npy")

    assert all(y_true_den == y_true_shear)
    y_true = y_true_den

    ids_tested = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/tested_ids.npy")

    FPR_both = np.zeros((50, 6))
    TPR_both = np.zeros((50, 6))
    AUC_both = []

    predictions = [y_predicted_shear, y_predicted_den]

    for i in range(2):
        y_predicted = predictions[i]
        FPR, TPR, AUC = get_roc_mass_bins(ids_tested, y_predicted, y_true)

        FPR_both[:, (3*i):((3*i)+3)] = FPR
        TPR_both[:, (3 * i):((3 * i) + 3)] =TPR
        AUC_both.append(AUC)

    plot_rocs_with_appropriate_legend_for_mass_bins(FPR_both, TPR_both, AUC_both)
    # plt.savefig("/Users/lls/Documents/CODE/stored_files/shear/classification/roc_mass_bins.pdf")
    plt.savefig("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol"
                "/rocs_mass_bins_den_plus_all_shear.pdf")


