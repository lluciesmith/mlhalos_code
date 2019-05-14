"""
Blind simulation:
ICs - /home/app/reseed/snapshot_099
Final snapshot - /home/app/reseed/IC.gadget3

"""

import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 18})
import numpy as np
from mlhalos import parameters
from mlhalos import machinelearning as ml
from scripts.ellipsoidal import predictions as ST_pred
from scripts.paper_plots import roc_plots
import matplotlib.pyplot as plt
from scripts.ellipsoidal import predictions as ell_pred


def particle_in_out_class(initial_parameters):
    """label in particles +1 and out particles -1."""

    id_to_h = initial_parameters.final_snapshot['grp']

    output = np.ones(len(id_to_h)) * -1
    output[np.where((id_to_h >= initial_parameters.min_halo_number) & (id_to_h <= initial_parameters.max_halo_number))] = 1
    output = output.astype("int")
    return output



############### ROC CURVES ###############

if __name__ == "__main__":
    # ic_training = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")
    # mass_threshold_in_out = ic_training.halo[400]['mass'].sum()

    ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/reseed50/IC.gadget3",
                                                final_snapshot="/Users/lls/Documents/CODE/reseed50/snapshot_099",
                                                load_final=True, min_halo_number=0, max_halo_number=400,
                                                min_mass_scale=3e10, max_mass_scale=1e15)

    # Change ic.ids_IN and ic.ids_OUT to be IN or OUT depending on whether they are in halos of mass larger than the
    # mass of halo 400 in ic_training and not that of halo 400 in ic. Change ids_IN, ids_OUT, and ic.max_halo_number.
    # We have that halo 409 in ic has the same mass as halo 400 in ic_training - hard code this for now.

    ic.max_halo_number = 409

    # Load or calculate true labels

    try:
        true_labels = np.load("/Users/lls/Documents/CODE/reseed50/predictions/true_labels.npy")
    except:
        print("Recalculating true labels")
        true_labels = particle_in_out_class(ic)
        np.save("/Users/lls/Documents/CODE/reseed50/predictions/true_labels.npy", true_labels)

    density_pred = np.load("/Users/lls/Documents/CODE/reseed50/predictions/density_predicted_probabilities.npy")
    density_shear_pred = np.load("/Users/lls/Documents/CODE/reseed50/predictions/shear_density_predicted_probabilities"
                                 ".npy")

    fpr_den, tpr_den, auc_den, thr = ml.roc(density_pred, true_labels)
    # fpr_shear_den, tpr_shear_den, auc_shear_den, thr_shear = ml.roc(density_shear_pred, true_labels)

    EPS_predicted_label = np.load("/Users/lls/Documents/CODE/reseed50/predictions/EPS_predicted_label.npy")
    ST_predicted_label = np.load("/Users/lls/Documents/CODE/reseed50/predictions/ST_predicted_label.npy")
    fpr_EPS, tpr_EPS = ST_pred.get_fpr_tpr_ellipsoidal_prediction(EPS_predicted_label, true_labels)
    fpr_ST, tpr_ST = ST_pred.get_fpr_tpr_ellipsoidal_prediction(ST_predicted_label, true_labels)

    # # ROC curves + EPS + ST for blind simulation
    #
    # # pred_all = np.array([density_pred, density_shear_pred])
    # # true_all = np.array([true_labels, true_labels])
    # #
    # # fig = roc_plots.get_multiple_rocs(pred_all, true_all, labels=["Density", "Density+Shear"],
    # #                   add_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS,
    # #                   add_ellipsoidal=True, fpr_ellipsoidal=fpr_ST, tpr_ellipsoidal=tpr_ST)
    # #
    # # plt.savefig("/Users/lls/Documents/CODE/reseed50/predictions/ROCs_blind_sim.pdf")
    #
    #
    # ######################## Fractional difference with training simulation ###########################
    #
    # # training data
    #
    # ic_training = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    #
    # pred_density_training = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/"
    #                        "predicted_den.npy")
    # true_density_training = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/true_den.npy")
    # fpr_den_training, tpr_den_training, auc_den_training, threshold = ml.roc(pred_density_training, true_density_training)
    # del pred_density_training
    # del true_density_training
    #
    # EPS_IN_label_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_IN.npy")
    # EPS_OUT_label_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_OUT.npy")
    # fpr_EPS_training, tpr_EPS_training = roc_plots.get_EPS_labels(EPS_IN_label_training, EPS_OUT_label_training)
    #
    # path = "/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol/"
    # pred_shear_training = np.load(path + "predicted_den+den_sub_ell+den_sub_prol.npy")
    # true_shear_training = np.load(path + "true_den+den_sub_ell+den_sub_prol.npy")
    # fpr_shear_training, tpr_shear_training, auc_shear_training, threshold = ml.roc(pred_shear_training,
    #                                                                          true_shear_training)
    #
    # den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")
    # fpr_ST_training, tpr_ST_training = ell_pred.get_fpr_tpr_from_features(den_f, mass_range="in",
    #                                                                       initial_parameters=ic_training,
    #                                                                       window_parameters=None,
    #                                                                       beta=0.485, gamma=0.615, a=0.707)
    #
    # figure, ax = plt.subplots(figsize=(8,6))
    # cols = ["#8856a7", "#7ea6ce"]
    #
    # ax.plot(fpr_den/fpr_den_training, tpr_den/tpr_den_training, lw=1.5, color=cols[0],
    #         label=r"$\mathrm{Density} (\Delta \mathrm{AUC} = $" + ' %.3f' % (auc_den - auc_den_training))
    # ax.plot(fpr_shear_den/fpr_shear_training, tpr_shear_den/tpr_shear_training, lw=1.5, color=cols[1],
    #         label=r"$\mathrm{Density+Shear} (\Delta \mathrm{AUC} = $" + ' %.3f' % (auc_den - auc_den_training))
    #
    # plt.scatter(fpr_EPS/fpr_EPS_training, tpr_EPS/tpr_EPS_training, color="k", s=30)
    # plt.scatter(fpr_ST/fpr_ST_training, tpr_ST/tpr_ST_training, color="k", marker="^", s=30)
    #
    # ax.set_xlabel(r'$\mathrm{FPR_{blind}/FPR_{training}}$', fontsize=20)
    # ax.set_ylabel(r'$\mathrm{TPR_{blind}/TPR_{training}}$', fontsize=20)
    #
    # plt.legend(loc="best", frameon=False)
    # plt.savefig("/Users/lls/Documents/CODE/reseed50/predictions/ROCs_differences.pdf")


