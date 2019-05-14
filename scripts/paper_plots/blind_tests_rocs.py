"""
Planck-like cosmology simulation pioneer50:
ICs - /share/data1/lls/pioneer50/pioneer50.512.ICs.tipsy
Final snapshot - /share/data1/lls/pioneer50/pioneer50.512.004096

"""

import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 18})
from mlhalos import machinelearning as ml
from scripts.ellipsoidal import predictions as ST_pred
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scripts.paper_plots import roc_plots as rp
from mlhalos import distinct_colours
import numpy as np


def particle_in_out_class(initial_parameters):
    """label in particles +1 and out particles -1."""

    id_to_h = initial_parameters.final_snapshot['grp']

    output = np.ones(len(id_to_h)) * -1
    output[np.where((id_to_h >= initial_parameters.min_halo_number) & (id_to_h <= initial_parameters.max_halo_number))] = 1
    output = output.astype("int")
    return output



############### ROC CURVES ###############

if __name__ == "__main__":

    ################ PLANCK SIMULATION ################

    planck_path = "/Users/lls/Documents/CODE/pioneer50/"

    planck_true_labels = np.load(planck_path + "predictions/true_labels.npy")
    planck_fpr_den = np.load(planck_path + "predictions/fpr_den.npy")
    planck_tpr_den = np.load(planck_path + "predictions/tpr_den.npy")
    planck_auc_den = trapz(planck_tpr_den, planck_fpr_den)

    planck_fpr_shear_den = np.load(planck_path + "predictions/fpr_shear_den.npy")
    planck_tpr_shear_den = np.load(planck_path + "predictions/tpr_shear_den.npy")
    planck_auc_shear_den = trapz(planck_tpr_shear_den, planck_fpr_shear_den)

    planck_EPS_predicted_label = np.load(planck_path + "predictions/EPS_predicted_label.npy")
    planck_fpr_EPS, planck_tpr_EPS = ST_pred.get_fpr_tpr_ellipsoidal_prediction(planck_EPS_predicted_label, planck_true_labels)

    planck_ST_predicted_label = np.load(planck_path + "predictions/ST_predicted_label.npy")
    planck_fpr_ST, planck_tpr_ST = ST_pred.get_fpr_tpr_ellipsoidal_prediction(planck_ST_predicted_label, planck_true_labels)


    ################# ORIGINAL SIMULATION ################

    original_path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"
    original_true_labels = np.load(original_path + "density_only/true_den.npy")

    orig_pred_density = np.load(original_path + "density_only/predicted_den.npy")
    orig_fpr, orig_tpr, orig_auc, orig_threshold = ml.roc(orig_pred_density, original_true_labels, true_class=1)

    orig_pred_shear = np.load(original_path + "den+den_sub_ell+den_sub_prol/predicted_den+den_sub_ell+den_sub_prol.npy")
    orig_fpr_shear, orig_tpr_shear, orig_auc_shear, orig_threshold_shear = ml.roc(orig_pred_shear, original_true_labels,
                                                                                  true_class=1)

    EPS_IN_label_orig = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_IN.npy")
    EPS_OUT_label_orig = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_OUT.npy")
    fpr_EPS_orig, tpr_EPS_orig = rp.get_EPS_labels(EPS_IN_label_orig, EPS_OUT_label_orig)

    fpr_ST_orig = np.load(original_path + "density_only/fpr_ST_prediction.npy")
    tpr_ST_orig = np.load(original_path + "density_only/tpr_ST_prediction.npy")

    ################# BLIND WMAP5 SIMULATION ################

    blind_path = "/Users/lls/Documents/CODE/reseed50/"
    true_labels_blind = np.load(blind_path + "predictions/true_labels.npy")

    density_pred_blind = np.load(blind_path + "predictions/density_predicted_probabilities.npy")
    fpr_den_blind, tpr_den_blind, auc_den_blind, thr_blind = ml.roc(density_pred_blind, true_labels_blind)

    density_shear_pred_blind = np.load(blind_path + "predictions/shear_density_predicted_probabilities.npy")
    fpr_shear_den_blind, tpr_shear_den_blind, auc_shear_den_blind, thr_shear_blind = ml.roc(density_shear_pred_blind,
                                                                                            true_labels_blind)

    EPS_predicted_label_blind = np.load("/Users/lls/Documents/CODE/reseed50/predictions/EPS_predicted_label.npy")
    ST_predicted_label_blind = np.load("/Users/lls/Documents/CODE/reseed50/predictions/ST_predicted_label.npy")
    fpr_EPS_blind, tpr_EPS_blind = ST_pred.get_fpr_tpr_ellipsoidal_prediction(EPS_predicted_label_blind, true_labels_blind)
    fpr_ST_blind, tpr_ST_blind = ST_pred.get_fpr_tpr_ellipsoidal_prediction(ST_predicted_label_blind, true_labels_blind)

    ################## PLOTS ################

    dc = distinct_colours.get_distinct(2)
    c_0 = "r"
    c_1 = "b"
    c_orig = "grey"
    s = 10

    figure, (ax, ax1) = plt.subplots(figsize=(8, 9), ncols=1, nrows=2, sharex=True, sharey=True)

    ax.plot(orig_fpr, orig_tpr, lw=1.5, color=c_orig, label="Training WMAP5 simulation")
    ax.plot(planck_fpr_den, planck_tpr_den, color=c_0,  lw=2.5, ls="--", label="Test Planck simulation")
    ax.plot(fpr_den_blind, tpr_den_blind, ls=":", color=c_1, lw=3, label="Test WMAP5 simulation")

    ax.plot(fpr_EPS_orig, tpr_EPS_orig, color=c_orig, linestyle="", marker="o", markersize=s, markeredgecolor=c_orig)
    ax.plot(planck_fpr_EPS, planck_tpr_EPS, color=c_0, linestyle="", marker="o", markersize=s, markeredgecolor=c_0)
    ax.plot(fpr_EPS_blind, tpr_EPS_blind, color=c_1, linestyle="", marker="o", markersize=s, markeredgecolor=c_1)

    ax1.plot(orig_fpr_shear, orig_tpr_shear, lw=1.5, color=c_orig)
    ax1.plot(planck_fpr_shear_den, planck_tpr_shear_den, color=c_0, lw=2.5, ls="--")
    ax1.plot(fpr_shear_den_blind, tpr_shear_den_blind, ls=":", color=c_1, lw=3)

    ax1.plot(planck_fpr_ST, planck_tpr_ST, color=c_0, linestyle="", marker="^", markersize=s, markeredgecolor=c_0)
    ax1.plot(fpr_ST_blind, tpr_ST_blind, color=c_1, linestyle="", marker="^", markersize=12, markeredgecolor=c_1)
    ax1.plot(fpr_ST_orig, tpr_ST_orig, color=c_orig, linestyle="", marker="^", markersize=10, markeredgecolor=c_orig)

    ax1.set_xlabel('False Positive Rate')
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylim(-0.03, 1.03)

    # LEGEND OPTION 1
    ax.legend(frameon=False, fontsize=16, fancybox=True, framealpha=0.5, bbox_to_anchor=(0.45, 0.2),
              loc="center")

    # LEGEND OPTION 2
    # legend1 = ax.legend(frameon=False, fontsize=16, fancybox=True, framealpha=0.5, bbox_to_anchor=(0.45, 0.2),
    #                     loc="center")
    # l2, = ax.plot([], [], color="k", linestyle="", marker="o", markersize=8, markeredgecolor="k")
    # legend2 = ax.legend([l2], ["EPS prediction"], frameon=False, fontsize=14, fancybox=True, framealpha=0.5,
    #                     bbox_to_anchor=(0.78, 0.6), loc="center", numpoints=1)
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)
    #
    # l3, = ax1.plot([], [], color="k", linestyle="", marker="^", markersize=8, markeredgecolor="k")
    # legend3 = ax1.legend([l3], ["ST prediction"], frameon=False, fontsize=14, fancybox=True, framealpha=0.5,
    #            bbox_to_anchor=(0.78, 0.6), loc="center",  numpoints=1)

    ax1.plot([], [], color="k", linestyle="", marker="^", markersize=8, markeredgecolor="k", label="ST prediction")
    ax1.plot([], [], color="k", linestyle="", marker="o", markersize=8, markeredgecolor="k", label="EPS prediction")
    ax1.legend(frameon=True, fontsize=16, fancybox=True, framealpha=0.5, bbox_to_anchor=(0.45, 0.2), numpoints=1,
              loc="center")

    ax.text(0.78, 0.7, "DENSITY\nFEATURES", fontsize=17, color="k", horizontalalignment='center',
            multialignment='center',
            fontweight="bold", transform=ax.transAxes)
    ax1.text(0.78, 0.7, "DENSITY+SHEAR\nFEATURES", fontsize=17, color="k", multialignment='center',
             horizontalalignment='center',fontweight="bold",
             transform=ax1.transAxes)
    figure.text(0.06, 0.55, 'True Positive Rate', fontsize=19, rotation='vertical', horizontalalignment='center',
                verticalalignment="center",
                multialignment='center')

    plt.subplots_adjust(hspace=0)
    plt.savefig("/Users/lls/Desktop/crop_legend.pdf")


    ################  other figure  ################

    # figure, (ax, ax1) = plt.subplots(figsize=(8, 6), ncols=1, nrows=2, sharex=True, sharey=True)
    # lw=2.5
    #
    # ax.plot(orig_fpr, orig_tpr, lw=1.5, color="#8856a7")
    # ax.plot(planck_fpr_den, planck_tpr_den, color="#8856a7",  lw=lw, ls="--")
    # ax.plot(fpr_den_blind, tpr_den_blind, ls="dotted", color="#8856a7", lw=lw)
    # ax.legend(loc="best", frameon=False)
    #
    # ax.set_xlim(-0.03, 1.03)
    # ax.set_ylim(-0.03, 1.03)
    # #plt.savefig("/Users/lls/Documents/CODE/pioneer50/predictions/ROCs_density_3_sims.png")
    #
    # ax1.plot(orig_fpr_shear, orig_tpr_shear, lw=1.5, color="#7ea6ce")
    # ax1.plot(planck_fpr_shear_den, planck_tpr_shear_den, color="#7ea6ce", lw=lw, ls="--")
    # ax1.plot(fpr_shear_den_blind, tpr_shear_den_blind, ls="dotted", color="#7ea6ce", lw=lw)
    #
    # s=20
    # ax.scatter(fpr_EPS_orig, tpr_EPS_orig, color="k", s=s)
    # ax.scatter(planck_fpr_EPS, planck_tpr_EPS, color="k", s=s)
    # ax.scatter(fpr_EPS_blind, tpr_EPS_blind, color="k", s=s)
    # ax1.scatter(fpr_ST_orig, tpr_ST_orig, color="k", s=s, marker="^")
    # ax1.scatter(planck_fpr_ST, planck_tpr_ST, color="k", s=s, marker="^")
    # ax1.scatter(fpr_ST_blind, tpr_ST_blind, color="k", s=s, marker="^")
    #
    # ax1.legend(loc="best", frameon=False, fontsize=15)
    # ax1.set_xlabel('False Positive Rate')
    # # ax1.set_ylabel('True Positive Rate')
    # ax1.set_xlim(-0.03, 1.03)
    # ax1.set_ylim(-0.03, 1.03)
    #
    # labels = ["Training WMAP5 sim", "Blind Planck sim", "Blind WMAP5 sim"]
    # l1 = ax.plot([], [], ls="-", lw=2, color="k", label=labels[0])
    # l2 = ax.plot([], [], ls="--", lw=2, color="k", label=labels[1])
    # l3 = ax.plot([], [], ls=":", lw=2, color="k", label=labels[2])
    # ax.legend(frameon=True, fontsize=15, fancybox=True, framealpha=0.5,  bbox_to_anchor=(0.5, 0.1), loc="center")
    # #ax.legend((l1, l2, l3), (labels[0], labels[1], labels[2]), loc="lower right")
    #
    # ax.text(0.8, 0.7, "DENSITY", fontsize=15, color="#8856a7",horizontalalignment='center', multialignment='center',
    #         fontweight="bold",
    #         transform=ax.transAxes)
    # ax1.text(0.8, 0.7, "DENSITY+SHEAR", fontsize=15, color="#7ea6ce", multialignment='center',
    #          horizontalalignment='center',fontweight="bold",
    #          transform=ax1.transAxes)
    # figure.text(0.06, 0.7, 'True Positive Rate', fontsize=18, rotation='vertical', multialignment='center')
    # plt.subplots_adjust(hspace=0)
    # #plt.subplots_adjust(top=0.9)
    # #plt.savefig("/Users/lls/Documents/CODE/pioneer50/predictions/ROCs_shear_density_3_sims.png")

    # ax.text(0.5, 0.3, "Training WMAP5 sim", fontsize=15, color=c_orig,horizontalalignment='center',
    #         multialignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.5, 0.2, "Blind Planck sim", fontsize=15, color=c_0,horizontalalignment='center',
    #         multialignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.5, 0.1, "Blind WMAP5 sim", fontsize=15, color=c_1,horizontalalignment='center',
    #         multialignment='center',
    #         transform=ax.transAxes)