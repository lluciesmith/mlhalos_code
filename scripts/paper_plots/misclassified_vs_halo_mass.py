import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import numpy as np
from mlhalos import parameters
import matplotlib.pyplot as plt
from mlhalos import distinct_colours


def get_testing_index(path="/Users/lls/Documents/CODE/stored_files/all_out/"):
    training_index = np.load(path + "50k_features_index.npy")
    testing_index = ~np.in1d(np.arange(256 ** 3), training_index)
    return testing_index


def get_ids_and_halos_test_set(path="/Users/lls/Documents/CODE/stored_files/all_out/"):
    # halos are ordered such that each halo mass corresponds to particles in order of particle ID (0,1,2,...)
    # Need to reorder the array such that it has first all IN particles, the all OUT particles

    testing_index = get_testing_index(path=path)

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    ids = np.concatenate((ic.ids_IN, ic.ids_OUT))
    ids_tested = ids[testing_index]

    halos = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy")
    halos_testing_particles = halos[ids_tested]

    return ids_tested, halos_testing_particles


# FALSE NEGATIVES FUNCTIONS


def get_false_negatives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    FNs = ids[~labels & y_bool]
    return FNs


def get_false_negatives_index(y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    ind = (~labels & y_bool)
    return ind


def false_negatives_ids_index_per_threshold(y_predicted, y_true, threshold_list):
    FNs = np.array([get_false_negatives_index(y_predicted, y_true, threshold=threshold_list[i])
                    for i in range(len(threshold_list))])
    return FNs


#  FALSE POSITIVES FUNCTIONS

def get_false_positives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    FPs = ids[labels & ~y_bool]
    return FPs


def get_false_positives_index(y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    ind = (labels & ~y_bool)
    return ind


def get_true_negatives_index(y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    ind = (~labels & ~y_bool)
    return ind


def true_negatives_ids_index_per_threshold(y_predicted, y_true, threshold_list):
    TNs = np.array([get_true_negatives_index(y_predicted, y_true, threshold=threshold_list[i])
                    for i in range(len(threshold_list))])
    return TNs


def false_positives_ids_index_per_threshold(y_predicted, y_true, threshold_list):
    FPs = np.array([get_false_positives_index(y_predicted, y_true, threshold=threshold_list[i])
                    for i in range(len(threshold_list))])
    return FPs


# binning

def bin_in_equal_number_of_halos(halo_mass_particles, bins_try=8):
    a = np.array_split(np.unique(halo_mass_particles), bins_try)
    bins = np.array([a[0].min()])
    bins = np.append(bins, np.array([a[i].max() for i in range(len(a))]))
    return bins


def poisson_propagated_errorbars(number, total_number):
    err = ((np.sqrt(number) * total_number) + (np.sqrt(total_number * number)))/total_number**2
    return err


def get_fraction_misclassfied_at_threshold_scale(false_positives_all_th, false_negatives_all_th, threshold,
                                                 halos_of_tested_particles, true_label, bins_out=None, bins_in=None):
    h_fps = halos_of_tested_particles[false_positives_all_th[threshold]]
    h_fns = halos_of_tested_particles[false_negatives_all_th[threshold]]

    if bins_out is None:
        bins_out = 20
    if bins_in is None:
        bins_in = bin_in_equal_number_of_halos(halos_of_tested_particles[true_label == 1])
        bins_in = np.log10(bins_in)

    n_total_out, bins_total_out = np.histogram(np.log10(halos_testing_particles[(true_label == -1) &
                                                                                (halos_of_tested_particles > 0)]),
                                               bins=bins_out)
    n_total_in, bins_total_in = np.histogram(np.log10(halos_of_tested_particles[true_label == 1]), bins=bins_in)

    FPs_n_den, bins1 = np.histogram(np.log10(h_fps[h_fps > 0]), bins=bins_total_out)
    FNs_n_den, bins2 = np.histogram(np.log10(h_fns), bins=bins_total_in)

    FPs_fraction = FPs_n_den/n_total_out
    FNs_fraction = FNs_n_den/ n_total_in

    FNs_yerr = poisson_propagated_errorbars(FNs_n_den, n_total_in)
    FPs_yerr = poisson_propagated_errorbars(FPs_n_den, n_total_out)

    return FPs_fraction, FNs_fraction, FPs_yerr, FNs_yerr, bins_total_in, bins_total_out


if __name__ == "__main__":

    path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"
    y_pred_den = np.load(path + "density_only/predicted_den.npy")
    y_true_den = np.load(path + "density_only/true_den.npy")

    # Find FPs and FNs of density run

    th = np.linspace(0, 1, 50)[::-1]
    ids_tested, halos_testing_particles = get_ids_and_halos_test_set()

    FPs_den_thr = false_positives_ids_index_per_threshold(y_pred_den, y_true_den, th)
    FNs_den_thr = false_negatives_ids_index_per_threshold(y_pred_den, y_true_den, th)
    FPs_28, FNs_28, FPs_yerr_28, FNs_yerr_28, b_in, b_out = \
        get_fraction_misclassfied_at_threshold_scale(FPs_den_thr, FNs_den_thr, 28, halos_testing_particles, y_true_den)

    FPs_24, FNs_24, FPs_yerr_24, FNs_yerr_24, b_in, b_out = \
        get_fraction_misclassfied_at_threshold_scale(FPs_den_thr, FNs_den_thr, 24, halos_testing_particles,
                                                     y_true_den, bins_in=b_in, bins_out=b_out)
    FPs_13, FNs_13, FPs_yerr_13, FNs_yerr_13, b_in, b_out = \
        get_fraction_misclassfied_at_threshold_scale(FPs_den_thr, FNs_den_thr, 13, halos_testing_particles, y_true_den,
                                                     bins_in=b_in, bins_out=b_out)

    # plot

    f, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    c = distinct_colours.get_distinct(2)
    color_13 = "#4d0000"
    color_24 = "#b30000"
    color_28 = "#ff6666"

    b_mid_out = (b_out[:-1] + b_out[1:])/2
    b_mid_in = ((b_in[:-1] + b_in[1:])/2)

    ax1.errorbar(b_mid_out, FPs_13, yerr=FPs_yerr_13, color=color_13, lw=1.5)
    ax1.errorbar(b_mid_in[~np.isinf(FNs_13)], FNs_13[~np.isinf(FNs_13)], yerr=FNs_yerr_13[~np.isinf(FNs_13)], color=color_13, lw=1.5)
    ax1.errorbar(b_mid_out, FPs_24, yerr=FPs_yerr_24, color=color_24, lw=1.5)
    ax1.errorbar(b_mid_in[~np.isinf(FNs_24)], FNs_24[~np.isinf(FNs_24)], yerr=FNs_yerr_24[~np.isinf(FNs_24)], color=color_24, lw=1.5)
    ax1.errorbar(b_mid_out, FPs_28, yerr=FPs_yerr_28, color=color_28, lw=1.5)
    ax1.errorbar(b_mid_in[~np.isinf(FNs_28)], FNs_28[~np.isinf(FNs_28)], yerr=FNs_yerr_28[~np.isinf(FNs_28)],
                 color=color_28, lw=1.5)

    ax1.set_ylim(0, 0.6)
    ax1.set_xlabel(r"$\log_{10}(M_{\mathrm{true}}/\mathrm{M}_{\odot})$", fontsize=20)
    ax1.set_ylabel(r"$N_{\mathrm{misclassified}}/ N_{\mathrm{all}}$", fontsize=20)
    ax1.set_xlim(10.4, 14.1)

    h_400_mass = 1836194204280.7886
    ax1.axvline(x=np.log10(h_400_mass), color="k", alpha=100, ls="--", lw=3)
    ax1.text(np.log10(h_400_mass) + 0.1, 0.1, "IN", fontweight="bold", fontsize=18, horizontalalignment="left",
           color="k", alpha=100,
           # transform=ax1.transAxes
           )
    ax1.text(np.log10(h_400_mass) - 0.1, 0.1, "OUT", fontweight="bold",fontsize=18, horizontalalignment="right",
           color="k", alpha=100,
           # transform=ax1.transAxes
           )

    plt.savefig("/Users/lls/Documents/mlhalos_paper/alternative_misclass_vs_halo_mass.pdf")

    # # Other version of plot with "particles in no halos" fraction
    #
    # f, (ax2, ax1) = plt.subplots(1, 2, sharey=True, gridspec_kw = {'width_ratios':[1.1, 8]}, figsize=(6.9, 5.2))
    #
    # c = distinct_colours.get_distinct(2)
    # color = c[0]
    #
    # b_mid_out = (bins_total_out[:-1] + bins_total_out[1:])/2
    # b_mid_in = ((bins_total_in[:-1] + bins_total_in[1:])/2)[n_total_in>0]
    #
    # #ax1.errorbar(b_mid_out, FPs_r_den_sub_ell, yerr=FPs_yerr, color="b", label="density+den-sub ellipticity")
    # ax1.errorbar(b_mid_out, FPs_fraction, yerr=FPs_yerr, color=color, label="density", lw=1.5)
    # # ax1.plot(b_mid_out, FPs_fraction, color=color, lw=1.5)
    # # ax1.scatter(b_mid_out, FPs_fraction, color=color)
    #
    # #ax1.errorbar(b_mid_in, FNs_r_den_sub_ell, yerr=FNs_yerr, color="b")
    # ax1.errorbar(b_mid_in, FNs_fraction, yerr=FNs_yerr, color=color, lw=1.5)
    # # ax1.plot(b_mid_in, FNs_fraction, color=color, lw=1.5)
    # # ax1.scatter(b_mid_in, FNs_fraction, color=color)
    #
    # ax1.set_ylim(0,0.6)
    #
    # ax2.axhline(out_out_fraction, color=color, lw=1.5)
    # # ax2.axhline(r_den_den_sub_ell_FPs_out_out, color="b")
    # ax2.set_xticks([])
    #
    # #ax1.axhline(0.5, color="k", ls="--", lw=1)
    # #ax2.axhline(0.5, color="k", ls="--", lw=1)
    # #f.text(0.65, 0.8445, "Random performance", fontsize=15, color="k", fontweight="bold")
    # #f.text(0.75, 0.51, "Random", multialignment='center', fontsize=15, color="k", fontweight="bold", xycoords="data")
    # #ax1.text(0.75, 0.49, "performance", multialignment='center', fontsize=15, color="k", fontweight="bold",
    # #       xycoords="data")
    #
    # # ax1.legend(loc="best", fontsize=16)
    # f.subplots_adjust(wspace=0.01)
    # ax1.set_xlabel(r"$\log_{10}(\mathrm{M_{true}}/\mathrm{M}_{\odot})$", fontsize=20)
    # ax2.set_ylabel(r"$N_{\mathrm{misclassified}}/ N_{\mathrm{all}}$", fontsize=20)
    # ax1.set_xlim(10.4, 14.1)
    #
    # h_400_mass = 1836194204280.7886
    # ax1.axvline(x=np.log10(h_400_mass), color="grey", alpha=100, ls="--", lw=3)
    # ax1.text(np.log10(h_400_mass) + 0.1, 0.1, "IN", fontweight="bold", fontsize=18, horizontalalignment="left",
    #        color="grey", alpha=100,
    #        # transform=ax1.transAxes
    #        )
    # ax1.text(np.log10(h_400_mass) - 0.1, 0.1, "OUT", fontweight="bold",fontsize=18, horizontalalignment="right",
    #        color="grey", alpha=100,
    #        # transform=ax1.transAxes
    #        )
    #
    # f.text(0.155, 0.87, "Not in\nhalos", multialignment='center', fontweight="bold",fontsize=15,
    #        color="k", transform = ax2.transAxes)
    # # plt.title("Threshold " + "%.3f" % th[scale])
    # plt.savefig("/Users/lls/Documents/mlhalos_paper/Figure_misclassified_vs_halo_mass.pdf")
