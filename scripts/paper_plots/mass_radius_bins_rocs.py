import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
import numpy as np
import matplotlib.pyplot as plt
from mlhalos import machinelearning as ml
from mlhalos import distinct_colours
from mlhalos import parameters
from matplotlib.ticker import MaxNLocator
from utils import mass as mass_util


if __name__ == "__main__":

    # MASS BINS CLASSIFICATION

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    high_halo = 6
    mid_halo = 78

    # Take classification results located at /Users/lls/Documents/CODE/stored_files/all_out and plot ROC curves for the
    # three mass bins separately

    results = np.load("/Users/lls/Documents/CODE/stored_files/all_out/classification_results.npy")
    # EPS_IN_label = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_IN.npy")
    # EPS_OUT_label = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_OUT.npy")
    # EPS_probabilities_all = np.concatenate((EPS_IN_label, EPS_OUT_label))
    EPS_probabilities_all = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predicted_labels.npy")

    ST_predicted_labels_all = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/ST_predicted_labels"
                                   ".npy")
    training_index = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy")
    testing_index = ~np.in1d(np.arange(256 ** 3), training_index)
    ST_predicted_labels = ST_predicted_labels_all[testing_index]
    del ST_predicted_labels_all

    # original

    fpr_orig, tpr_orig, auc_orig, threshold = ml.roc(results[:,2:4], results[:,1])

    # # small bin
    #
    # fpr_small, tpr_small, auc_small = clm.get_fpr_tpr_auc_mass_bin(ic, mass_bin="small",
    #                                                            high_halo=6, mid_halo=78, results=results)
    #
    # # mid bin
    #
    # fpr_mid, tpr_mid, auc_mid = clm.get_fpr_tpr_auc_mass_bin(ic, mass_bin="mid", high_halo=6, mid_halo=78,
    #                                                          results=results)
    #
    # # high bin
    #
    # fpr_high, tpr_high, auc_high = clm.get_fpr_tpr_auc_mass_bin(ic, mass_bin="high", high_halo=6, mid_halo=78,
    #                                                         results=results)

    fpr_small, tpr_small, auc_small = mass_util.get_fpr_tpr_auc_mass_bin(results, mass_bin="small",
                                                               high_halo=6, mid_halo=78)
    fpr_EPS_small, tpr_EPS_small = mass_util.get_EPS_fpr_tpr_auc_mass_bin(results, EPS_probabilities_all,
                                                                          mass_bin="small", high_halo=6, mid_halo=78)
    fpr_ST_small, tpr_ST_small = mass_util.get_EPS_fpr_tpr_auc_mass_bin(results, ST_predicted_labels,
                                                                          mass_bin="small", high_halo=6, mid_halo=78)

    # mid bin

    fpr_mid, tpr_mid, auc_mid = mass_util.get_fpr_tpr_auc_mass_bin(results, mass_bin="mid", high_halo=6, mid_halo=78)
    fpr_EPS_mid, tpr_EPS_mid = mass_util.get_EPS_fpr_tpr_auc_mass_bin(results, EPS_probabilities_all,
                                                                       mass_bin="mid", high_halo=6, mid_halo=78)
    fpr_ST_mid, tpr_ST_mid = mass_util.get_EPS_fpr_tpr_auc_mass_bin(results, ST_predicted_labels,
                                                                          mass_bin="mid", high_halo=6, mid_halo=78)

    # high bin

    fpr_high, tpr_high, auc_high = mass_util.get_fpr_tpr_auc_mass_bin(results, mass_bin="high", high_halo=6,
                                                                      mid_halo=78)
    fpr_EPS_high, tpr_EPS_high = mass_util.get_EPS_fpr_tpr_auc_mass_bin(results, EPS_probabilities_all,
                                                                        mass_bin="high", high_halo=6, mid_halo=78)
    fpr_ST_high, tpr_ST_high = mass_util.get_EPS_fpr_tpr_auc_mass_bin(results, ST_predicted_labels,
                                                                          mass_bin="high", high_halo=6, mid_halo=78)



    # RADIUS BINS CLASSIFICATION

    radii_properties_in = np.load(
        "/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_in_ids.npy")
    radii_properties_out = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_out_ids"
                                   ".npy")

    fraction = np.concatenate((radii_properties_in[:, 2], radii_properties_out[:, 2]))
    radius = np.concatenate((radii_properties_in[:, 1], radii_properties_out[:, 1]))
    ids_in_halo = np.concatenate((radii_properties_in[:, 0], radii_properties_out[:, 0]))

    f = ic.final_snapshot
    ids_no_halo = f['iord'][f['grp'] == -1]

    # 30% radius

    # ids_30_in_halo = ids_in_halo[(fraction <= 0.3) & (radius > 25.6)]
    ids_30_in_halo = ids_in_halo[(fraction <= 0.3) & (radius > 25.6)]
    ids_30 = np.concatenate((ids_30_in_halo, ids_no_halo))

    ids_30_classified = results[:, 0][np.in1d(results[:, 0], ids_30)]
    true_label_ids_30 = results[:, 1][np.in1d(results[:, 0], ids_30_classified)]
    predicted_probabilities_ids_30 = results[:, 2:4][np.in1d(results[:, 0], ids_30_classified)]
    fpr_30, tpr_30, auc_30, threshold = ml.roc(predicted_probabilities_ids_30, true_label_ids_30)

    fpr_EPS_30, tpr_EPS_30, auc_EPS_30, threshold_30 = ml.roc(EPS_probabilities_all[np.in1d(results[:, 0],ids_30_classified)],true_label_ids_30)
    fpr_EPS_30 = fpr_EPS_30[0]
    tpr_EPS_30 = tpr_EPS_30[0]

    fpr_ST_30, tpr_ST_30, auc_ST_30, threshold_30 = ml.roc(ST_predicted_labels[np.in1d(results[:, 0],
                                                                                      ids_30_classified)],
                                                              true_label_ids_30)
    fpr_ST_30 = fpr_ST_30[0]
    tpr_ST_30 = tpr_ST_30[0]

    # 30-60% radius

    ids_30_60_in_halo = ids_in_halo[(fraction > 0.3) & (fraction < 0.6) & (radius > 25.6)]
    ids_30_60 = np.concatenate((ids_30_60_in_halo, ids_no_halo))

    ids_30_60_classified = results[:, 0][np.in1d(results[:, 0], ids_30_60)]
    true_label_ids_30_60 = results[:, 1][np.in1d(results[:, 0], ids_30_60_classified)]
    predicted_probabilities_ids_30_60 = results[:, 2:4][np.in1d(results[:, 0], ids_30_60_classified)]
    fpr_30_60, tpr_30_60, auc_30_60, threshold = ml.roc(predicted_probabilities_ids_30_60, true_label_ids_30_60)

    fpr_EPS_30_60, tpr_EPS_30_60, auc_EPS_30_60, threshold_30_60 = ml.roc(EPS_probabilities_all[np.in1d(results[:, 0],
                                                                                            ids_30_60_classified)],
                                                                          true_label_ids_30_60)
    fpr_EPS_30_60 = fpr_EPS_30_60[0]
    tpr_EPS_30_60 = tpr_EPS_30_60[0]

    fpr_ST_30_60, tpr_ST_30_60, auc_ST_30_60, threshold_30_60 = ml.roc(ST_predicted_labels[np.in1d(results[:, 0],
                                                                                       ids_30_60_classified)],
                                                                       true_label_ids_30_60)
    fpr_ST_30_60 = fpr_ST_30_60[0]
    tpr_ST_30_60 = tpr_ST_30_60[0]

    # 60-100% radius

    ids_60_100_in_halo = ids_in_halo[(fraction > 0.6) & (fraction < 1) & (radius > 25.6)]
    ids_60_100 = np.concatenate((ids_60_100_in_halo, ids_no_halo))

    ids_60_100_classified = results[:, 0][np.in1d(results[:, 0], ids_60_100)]
    true_label_ids_60_100 = results[:, 1][np.in1d(results[:, 0], ids_60_100_classified)]
    predicted_probabilities_ids_60_100 = results[:, 2:4][np.in1d(results[:, 0], ids_60_100_classified)]
    fpr_60_100, tpr_60_100, auc_60_100, threshold = ml.roc(predicted_probabilities_ids_60_100, true_label_ids_60_100)

    fpr_EPS_60_100, tpr_EPS_60_100, auc_EPS_60_100, threshold_60_100= ml.roc(EPS_probabilities_all
                                                                          [np.in1d(results[:, 0], ids_60_100_classified)],
                                                                          true_label_ids_60_100)
    fpr_EPS_60_100 = fpr_EPS_60_100[0]
    tpr_EPS_60_100 = tpr_EPS_60_100[0]

    fpr_ST_60_100, tpr_ST_60_100, auc_ST_60_100, threshold_60_100 = ml.roc(ST_predicted_labels
                                                                       [np.in1d(results[:, 0], ids_60_100_classified)],
                                                                       true_label_ids_60_100)
    fpr_ST_60_100 = fpr_ST_60_100[0]
    tpr_ST_60_100 = tpr_ST_60_100[0]


    # PLOT MASS BINS + RADIUS BINS ROC CURVES

    high_mass = ic.halo[high_halo]['mass'].sum()
    mid_mass = ic.halo[mid_halo]['mass'].sum()

    figure, (ax1, ax) = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(15, 4.5))
    col = distinct_colours.get_distinct(3)

    col_high = "#6600cc"
    col_mid = "#003cb4"
    col_small = "#00b3f1"
    ms = 8

    ax.plot(fpr_orig, tpr_orig, color="grey", ls='--', lw=1.5)
    ax.plot(fpr_high, tpr_high, color=col_high, lw=1.5,
            label="Clusters (AUC = " + str(float('%.3g' % auc_high)) + ")")
    ax.plot(fpr_mid, tpr_mid, color=col_mid, lw=1.5,
            label="Groups (AUC = " + str(float('%.3g' % auc_mid))+ ")")
    ax.plot(fpr_small, tpr_small, color=col_small, lw=1.5,
            label="Galaxies (AUC = " + str(float('%.3g' % auc_small)) + ")")

    ax.plot(fpr_EPS_high, tpr_EPS_high, linestyle="", marker="o", markersize=ms, markeredgecolor=col_high, color=col_high)
    ax.plot(fpr_ST_high, tpr_ST_high, linestyle="", marker="^", markersize=ms, markeredgecolor=col_high, color=col_high)

    ax.plot(fpr_EPS_mid, tpr_EPS_mid, linestyle="", marker="o", markersize=ms, markeredgecolor=col_mid, color=col_mid)
    ax.plot(fpr_ST_mid, tpr_ST_mid, linestyle="", marker="^", markersize=ms, markeredgecolor=col_mid, color=col_mid)

    ax.plot(fpr_EPS_small, tpr_EPS_small, linestyle="", marker="o", markersize=ms, markeredgecolor=col_small, color=col_small)
    ax.plot(fpr_ST_small, tpr_ST_small, linestyle="", marker="^", markersize=ms, markeredgecolor=col_small, color=col_small)

    ax.legend(loc=(0.35, 0.1), fontsize=15,
              frameon=False)
    ax.set_xlabel('False Positive Rate', fontsize=17)
    ax1.set_ylabel('True Positive Rate', fontsize=17)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    col_30 = "#333300"
    col_60 = "#669900"
    col_100 = "#e69900"
    ax1.plot(fpr_orig, tpr_orig, color="grey", ls='--', lw=1.5)

    ax1.plot(fpr_30, tpr_30, color=col_30,lw=1.5, label="Inner (AUC = " + '%.3f' % auc_30 + ")")
    ax1.plot(fpr_30_60, tpr_30_60, color=col_60, lw=1.5, label="Mid (AUC = " + '%.3f' % auc_30_60 + ")")
    ax1.plot(fpr_60_100, tpr_60_100, color=col_100, lw=1.5, label="Outer (AUC = " + '%.3f' % auc_60_100 + ")")

    ax1.plot(fpr_EPS_30, tpr_EPS_30, linestyle="", marker="o", markersize=ms, markeredgecolor=col_30, color=col_30)
    ax1.plot(fpr_ST_30, tpr_ST_30, linestyle="", marker="^", markersize=ms, markeredgecolor=col_30, color=col_30)

    ax1.plot(fpr_EPS_30_60, tpr_EPS_30_60, linestyle="", marker="o", markersize=ms, markeredgecolor=col_60, color=col_60)
    ax1.plot(fpr_ST_30_60, tpr_ST_30_60, linestyle="", marker="^", markersize=ms, markeredgecolor=col_60, color=col_60)

    ax1.plot(fpr_EPS_60_100, tpr_EPS_60_100, linestyle="", marker="o", markersize=ms, markeredgecolor=col_100, color=col_100)
    ax1.plot(fpr_ST_60_100, tpr_ST_60_100, linestyle="", marker="^", markersize=ms, markeredgecolor=col_100, color=col_100)

    ax1.set_xlabel('False Positive Rate', fontsize=17)
    ax1.legend(loc=(0.35, 0.1), fontsize=15,
               #bbox_to_anchor=(0.95, 0.05),
               frameon=False)

    plt.subplots_adjust(wspace=0)
    nbins = len(ax.get_yticks())  # added
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))

    ax.text(0.37, 0.35, 'MASS RANGE', fontsize=15)
    ax1.text(0.37, 0.35, 'RADIAL RANGE', fontsize=15)

    ax.text(0.37, 0.35, 'MASS RANGE', fontsize=15)
    ax1.text(0.37, 0.35, 'RADIAL RANGE', fontsize=15)


    plt.savefig("/Users/lls/Documents/mlhalos_paper/Figure_mass_radius_roc_2.pdf")
    # plt.savefig("/Users/lls/Documents/mlhalos_paper/Figure_mass_radius_roc_with_EPS.pdf")

