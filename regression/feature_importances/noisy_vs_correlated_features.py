"""


"""
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from regression.feature_importances import functions_imp_tests as it
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from regression.plots import plotting_functions as pf


def violin_plots_density_vs_shear(shear_predicted, shear_true, density_predicted, density_true, bins, path=None,
                                  label_shear="den+shear", label_den="density"):

    shear_pred, shear_mean = pf.get_predicted_masses_in_each_true_m_bin(bins, shear_predicted, shear_true,
                                                                        return_mean=False)
    den_pred, den_mean = pf.get_predicted_masses_in_each_true_m_bin(bins, density_predicted, density_true,
                                                                    return_mean=False)

    width_xbins = np.diff(bins)
    xaxis = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(nrows=1, ncols=1)
    color = "b"
    vplot = axes.violinplot(shear_pred, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color) for b in vplot['bodies']]
    axes.errorbar(xaxis, shear_mean, xerr=width_xbins / 2, color="b", fmt="o", label=label_shear)

    vplot1 = axes.violinplot(den_pred, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color("r") for b in vplot1['bodies']]
    axes.errorbar(xaxis, den_mean, xerr=width_xbins / 2, color="r", fmt="o", label=label_den)

    axes.plot(bins, bins, color="k")
    axes.set_xlim(bins.min() - 0.1, bins.max() + 0.1)
    # axes.set_ylim(bins.min() - 0.1, bins.max() + 0.1)

    axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    axes.legend(loc=2)
    if path is not None:
        plt.savefig(path + "violins_shear_vs_den.pdf")


if __name__ == "__main__":


    # First load the training set and testing set to use for the feature importances tests

    training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/training_ids.npy")
    log_mass_training = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                                "files/log_halo_mass_training.npy")

    testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/testing_ids.npy")
    log_halo_mass_testing = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                                    "files/log_halo_mass_testing.npy")


    ############### Compare 0.4 correlated and 0.7 correlated ###############


    # training features
    dup = np.copy(log_mass_training)
    dup1 = np.tile(dup, (50, 1)).transpose()

    noise_07 = np.random.normal(0, 1.2, [len(log_mass_training), 50])
    signal_07_corr = dup1 + noise_07

    noise_04 = np.random.normal(0, 2.7, [len(log_mass_training), 50])
    signal_04_corr = dup1 + noise_04

    training_features_04_only = np.column_stack((signal_04_corr, log_mass_training))
    training_features_all = np.column_stack((signal_07_corr, signal_04_corr, log_mass_training))
    corrcoef_04_only = np.corrcoef(training_features_04_only.transpose())
    corrcoef_all = np.corrcoef(training_features_all.transpose())

    # testing features
    testing_dup = np.copy(log_halo_mass_testing)
    testing_dup1 = np.tile(testing_dup, (50, 1)).transpose()

    testing_noise_07 = np.random.normal(0, 1.2, [len(log_halo_mass_testing), 50])
    testing_signal_07_corr = testing_dup1 + testing_noise_07

    testing_noise_04 = np.random.normal(0, 2.7, [len(log_halo_mass_testing), 50])
    testing_signal_04_corr = testing_dup1 + testing_noise_04

    testing_features_07_04 = np.column_stack((testing_signal_07_corr, testing_signal_04_corr))

    # predictions
    pred_04_only, RF_04_only = it.train_and_test_algorithm(training_features_04_only, testing_signal_04_corr,
                                                           n_estimators=500, max_features=10, min_samples_leaf=1,
                                                           min_samples_split=2)

    pred_07_04_features, RF_07_04_features = it.train_and_test_algorithm(training_features_all, testing_features_07_04,
                                                                         n_estimators=500, max_features=10,
                                                                         min_samples_leaf=1, min_samples_split=2)

    # PLOT IMPORTANCES

    # 0.7 + 0.4 case

    fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), sharex=True)
    ax1.bar(range(50), RF_07_04_features.feature_importances_[:50], label="$0.7$ correlation", color="g")
    ax2.bar(range(50), RF_07_04_features.feature_importances_[50:], label="$0.4$ correlation", color="b")
    ax2.set_xlabel("Features")
    #ax2.set_ylabel("Importance")
    fig2.text(0.06, 0.6, 'Importance', fontsize=19, rotation='vertical')
    # fig2.title("$0.7 + 0.4$ correlation features")

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig2.subplots_adjust(hspace=0)

    y_max = RF_07_04_features.feature_importances_.max()
    length_ticks = 4
    yticks_change = np.linspace(0, y_max, num=length_ticks, endpoint=True)
    ax1.set_yticks(yticks_change)
    # yticks_change = np.linspace(0, 0.025, num=length_ticks, endpoint=True)
    ax2.set_yticks(yticks_change)

    nbins = length_ticks  # added
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/07_and_04_corr_importances.png")

    # 0.4 only case

    plt.figure()
    plt.bar(range(50), RF_04_only.feature_importances_, label="$0.4$ correlation", color="b")
    plt.legend(loc="best")
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/04_only_corr.png")


    # PLOT PREDICTIONS AS SCATTER PLOT

    # plot predictions

    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), sharey=True)
    ax1.scatter(log_halo_mass_testing, pred_04_only, alpha=0.1, label="$0.4$ corr features", color="b")
    ax2.scatter(log_halo_mass_testing, pred_07_04_features, alpha=0.1, label="$0.7$ and $0.4$ corr features", color="g")
    ax1.set_ylabel("Predicted (log) halo mass")
    ax1.set_xlabel("True (log) halo mass")
    ax2.set_xlabel("True (log) halo mass")

    ax1.legend(loc="best")
    ax2.legend(loc="best")

    plt.subplots_adjust(bottom=0.15,wspace=0)

    # PLOT PREDICTIONS AS VIOLIN PLOTS

    bins_plotting = np.linspace(log_halo_mass_testing.min(), log_halo_mass_testing.max(), 15, endpoint=True)
    violin_plots_density_vs_shear(pred_07_04_features, log_halo_mass_testing, pred_04_only, log_halo_mass_testing,
                                  bins_plotting, path=None, label_shear="$0.7 + 0.4$ correlation features",
                                  label_den="$0.4$ correlation features")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/violin_plots_07_04_corr_features.png")
