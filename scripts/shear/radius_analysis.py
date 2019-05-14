import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code')
from utils import classification_results as cres
from mlhalos import machinelearning as ml
from mlhalos import distinct_colours


def plot_rocs(fpr, tpr, ls="-", cols="b"):
    l, = plt.plot(fpr, tpr, lw=1.5, ls=ls, color=cols)
    return l,


def get_legend_radius_bins(a, b, c, d, g, h, cols, AUC_both):
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
                         [categories[0]] + [r"30 \%" + ' (%.3f)' % (AUC_both[0][0]), r"30\% - 60\%" + ' (%.3f)' % (
                             AUC_both[0][1]), r"60\% - 100\%" + ' (%.3f)' % (AUC_both[0][2])]
                         + [categories[1]] + [r"30\%"+ ' (%.3f)' % (AUC_both[1][0]), r"30\% - 60\%" + ' (%.3f)' % (
                             AUC_both[1][1]), r"60\% - 100\%" + ' (%.3f)' % (AUC_both[1][2])],
                         loc="best", ncol=2, fontsize=18)
    return legend1


def plot_rocs_with_appropriate_legend_for_radius_bins(FPR_both, TPR_both, AUC_both, figsize=(11,7)):
    figure, ax = plt.subplots(figsize=figsize)
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

    legend1 = get_legend_radius_bins(a, b, c, d, g, h, cols, AUC_both)
    plt.gca().add_artist(legend1)


def load_roc_subset_particles(ids_all, ids_subset, y_predicted, y_true):
    pred = y_predicted[np.in1d(ids_all, ids_subset)]
    true = y_true[np.in1d(ids_all, ids_subset)]
    fpr, tpr, auc, thr = ml.roc(pred, true)
    return fpr, tpr, auc, thr


if __name__ == "__main__":


    y_predicted_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/predicted_den+den_sub_ell.npy")
    y_true_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/true_den+den_sub_ell.npy")

    y_predicted_den = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/predicted_den.npy")
    y_true_den = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/true_den.npy")

    assert all(y_true_den == y_true_shear)
    y_true = y_true_den

    testing_index = np.load("/Users/lls/Documents/CODE/stored_files/testing_index.npy")
    ids_tested = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/tested_ids.npy")


    # Take classification results located at /Users/lls/Documents/CODE/stored_files/all_out and plot ROC curves for the
    # three mass bins separately

    radii_properties_in = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_in_ids.npy")
    radii_properties_out = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_out_ids"
                                   ".npy")

    fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
    radius = np.concatenate((radii_properties_in[:,1],radii_properties_out[:,1]))
    ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))

    f, h = cres.load_final_snapshot_and_halos()
    ids_no_halo = f['iord'][f['grp'] == -1]


    ############# 30% radius #############
    # ids_30_in_halo = ids_in_halo[(fraction < 0.3) & (radius > 25.6)]
    ids_30_in_halo = ids_in_halo[(fraction < 0.3)]
    ids_30 = np.concatenate((ids_30_in_halo, ids_no_halo))

    # density

    den_fpr_30, den_tpr_30, den_auc_30, threshold = load_roc_subset_particles(ids_tested, ids_30, y_predicted_den,
                                                                              y_true_den)

    # density+shear

    shear_fpr_30, shear_tpr_30, shear_auc_30, threshold = load_roc_subset_particles(ids_tested, ids_30,
                                                                                    y_predicted_shear, y_true_shear)


    ############# 30-60% radius #############
    # ids_30_60_in_halo = ids_in_halo[(fraction > 0.3) & (fraction < 0.6) & (radius > 25.6)]
    ids_30_60_in_halo = ids_in_halo[(fraction > 0.3) & (fraction < 0.6)]
    ids_30_60 = np.concatenate((ids_30_60_in_halo, ids_no_halo))

    # density

    den_fpr_30_60, den_tpr_30_60, den_auc_30_60, threshold = load_roc_subset_particles(ids_tested, ids_30_60,
                                                                                       y_predicted_den, y_true_den)

    # density+shear

    shear_fpr_30_60, shear_tpr_30_60, shear_auc_30_60, threshold = load_roc_subset_particles(ids_tested, ids_30_60,
                                                                                       y_predicted_shear, y_true_shear)


    ############# 60-100% radius  #############
    # ids_60_100_in_halo = ids_in_halo[(fraction > 0.6) & (fraction < 1) & (radius > 25.6)]
    ids_60_100_in_halo = ids_in_halo[(fraction > 0.6) & (fraction < 1)]
    ids_60_100 = np.concatenate((ids_60_100_in_halo, ids_no_halo))

    # density

    den_fpr_60_100, den_tpr_60_100, den_auc_60_100, threshold = load_roc_subset_particles(ids_tested, ids_60_100,
                                                                                          y_predicted_den, y_true_den)

    # density+shear

    shear_fpr_60_100, shear_tpr_60_100, shear_auc_60_100, threshold = load_roc_subset_particles(ids_tested, ids_60_100,
                                                                                       y_predicted_shear, y_true_shear)

    ############# original #############

    fpr_den, tpr_den, auc_den, threshold = ml.roc(y_predicted_den, y_true_den)
    fpr_shear, tpr_shear, auc_shear, threshold = ml.roc(y_predicted_shear, y_true_shear)


    ######### PLOT ########

    FPR_all = np.column_stack((shear_fpr_30, shear_fpr_30_60, shear_fpr_60_100, den_fpr_30, den_fpr_30_60, den_fpr_60_100))
    TPR_all = np.column_stack((shear_tpr_30, shear_tpr_30_60, shear_tpr_60_100, den_tpr_30, den_tpr_30_60, den_tpr_60_100))
    AUC_all = [[shear_auc_30, shear_auc_30_60, shear_auc_60_100],[den_auc_30, den_auc_30_60, den_auc_60_100] ]

    plot_rocs_with_appropriate_legend_for_radius_bins(FPR_all, TPR_all, AUC_all)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/shear/classification/rocs_radii_bins_ignoring_softening_length"
                ".pdf")
