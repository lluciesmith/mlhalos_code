"""
Planck-like cosmology simulation pioneer50:
ICs - /share/data1/lls/pioneer50/pioneer50.512.ICs.tipsy
Final snapshot - /share/data1/lls/pioneer50/pioneer50.512.004096

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
from scipy.integrate import trapz
from mlhalos import plot



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

    # path = "/share/data1/lls/pioneer50/"
    path = "/Users/lls/Documents/CODE/pioneer50/"
    # ic = parameters.InitialConditionsParameters(initial_snapshot=path + "pioneer50.512.ICs.tipsy",
    #                                             final_snapshot=path + "pioneer50.512.004096",
    #                                             load_final=True, min_halo_number=1, max_halo_number=134,
    #                                             min_mass_scale=3e10, max_mass_scale=1e15, sigma8=0.831)

    # Change ic.ids_IN and ic.ids_OUT to be IN or OUT depending on whether they are in halos of mass larger than the
    # mass of halo 400 in ic_training and not that of halo 400 in ic. Change ids_IN, ids_OUT, and ic.max_halo_number.
    # We have that halo 409 in ic has the same mass as halo 400 in ic_training - hard code this for now.

    # Load or calculate true labels

    try:
        true_labels = np.load(path + "predictions/true_labels.npy")
        print("Using saved true labels")
    except IOError:
        print("Calculating true labels")
        ic = parameters.InitialConditionsParameters(initial_snapshot=path + "pioneer50.512.ICs.tipsy",
                                                    final_snapshot=path + "pioneer50.512.004096",
                                                    load_final=True, min_halo_number=1, max_halo_number=134,
                                                    min_mass_scale=5e9, max_mass_scale=1e15, sigma8=0.831,
                                                    path="/Users/lls/Documents/CODE/")
        true_labels = particle_in_out_class(ic)
        np.save(path + "predictions/true_labels.npy", true_labels)

    try:
        fpr_den = np.load(path + "predictions/fpr_den.npy")
        tpr_den = np.load(path + "predictions/tpr_den.npy")
        auc_den = trapz(tpr_den, fpr_den)
    except IOError:
        density_pred = np.load(path + "predictions/density_predicted_probabilities.npy")
        density_shear_pred = np.load(path + "predictions/shear_density_predicted_probabilities.npy")
        fpr_den, tpr_den, auc_den, thr = ml.roc(density_pred, true_labels)
        print(auc_den)
        np.save(path + "predictions/fpr_den.npy", fpr_den)
        np.save(path + "predictions/tpr_den.npy", tpr_den)

    try:
        fpr_shear_den = np.load(path + "predictions/fpr_shear_den.npy")
        tpr_shear_den = np.load(path + "predictions/tpr_shear_den.npy")
        auc_shear_den = trapz(tpr_shear_den, fpr_shear_den)
    except IOError:
        density_shear_pred = np.load(path + "predictions/shear_density_predicted_probabilities.npy")
        fpr_shear_den, tpr_shear_den, auc_shear_den, thr_shear = ml.roc(density_shear_pred, true_labels)
        print(auc_shear_den)
        np.save(path + "predictions/fpr_shear_den.npy", fpr_shear_den)
        np.save(path + "predictions/tpr_shear_den.npy", tpr_shear_den)

    EPS_predicted_label = np.load(path + "predictions/EPS_predicted_label.npy")
    fpr_EPS, tpr_EPS = ST_pred.get_fpr_tpr_ellipsoidal_prediction(EPS_predicted_label, true_labels)
    print("EPS fpr is " + str(fpr_EPS) + " and tpr is " + str(tpr_EPS))

    ST_predicted_label = np.load(path + "predictions/ST_predicted_label.npy")
    fpr_ST, tpr_ST = ST_pred.get_fpr_tpr_ellipsoidal_prediction(ST_predicted_label, true_labels)
    print("ST fpr is " + str(fpr_ST) + " and tpr is " + str(tpr_ST))

    # ROC curves + EPS + ST for blind simulation

    FPR_all = np.array([fpr_den, fpr_shear_den])
    TPR_all = np.array([tpr_den, tpr_shear_den])
    auc_all = np.array([auc_den, auc_shear_den])

    fig = plot.roc_plot(FPR_all.transpose(), TPR_all.transpose(), auc_all, labels=["Density", "Density+Shear"],
                        add_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS,
                        add_ellipsoidal=True, fpr_ellipsoidal=fpr_ST, tpr_ellipsoidal=tpr_ST, frameon=False,
                        fontsize_labels=17, cols=["#8856a7", "#7ea6ce" ])

    plt.savefig(path + "predictions/ROCs_pioneer50_sim.pdf")

    ####### COMPARE WITH OTHER SIMS

    #original sim

    pred_density = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/"
                           "predicted_den.npy")
    true_density = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/true_den.npy")
    fpr_dOr, tpr_dOr, auc_dOr, threshold_dOr = ml.roc(pred_density, true_density, true_class=1)

    pred_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol/predicted_den+den_sub_ell+den_sub_prol.npy")
    true_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol/true_den+den_sub_ell+den_sub_prol.npy")
    fpr_dsOr, tpr_dsOr, auc_dsOr, threshold_dsOr = ml.roc(pred_shear, true_shear, true_class=1)

    plot.roc_plot(FPR_all.transpose(), TPR_all.transpose(), auc_all, labels=["Density", "Density+Shear"],
                  #add_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS,
                  #add_ellipsoidal=True, fpr_ellipsoidal=fpr_ST, tpr_ellipsoidal=tpr_ST,
                  frameon=False, fontsize_labels=17, cols=["#8856a7", "#7ea6ce"])
    plt.plot(fpr_dOr, tpr_dOr, ls="--", color="#8856a7")
    plt.plot(fpr_dsOr, tpr_dsOr, ls="--", color="#7ea6ce", lw=1.5)

    # blind

    density_pred_blind = np.load("/Users/lls/Documents/CODE/reseed50/predictions/density_predicted_probabilities.npy")
    density_shear_pred_blind = np.load("/Users/lls/Documents/CODE/reseed50/predictions/"
                                       "shear_density_predicted_probabilities.npy")
    true_labels_blind = np.load("/Users/lls/Documents/CODE/reseed50/predictions/true_labels.npy")

    fpr_den_blind, tpr_den_blind, auc_den_blind, thr_blind = ml.roc(density_pred_blind, true_labels_blind)
    fpr_shear_den_blind, tpr_shear_den_blind, auc_shear_den_blind, thr_shear_blind = ml.roc(density_shear_pred_blind, true_labels_blind)

    plot.roc_plot(FPR_all.transpose(), TPR_all.transpose(), auc_all, labels=["Density", "Density+Shear"],
                  #add_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS,
                  #add_ellipsoidal=True, fpr_ellipsoidal=fpr_ST, tpr_ellipsoidal=tpr_ST,
                  frameon=False, fontsize_labels=17, cols=["#8856a7", "#7ea6ce"])

    plt.plot(fpr_den, tpr_den, color="#8856a7",  label="Planck", lw=2)
    plt.plot(fpr_dOr, tpr_dOr, ls="--", color="#8856a7", label="orig", lw=2)
    plt.plot(fpr_den_blind, tpr_den_blind, ls="-.", color="#8856a7", label="blind wmap", lw=2)
    plt.legend(loc="best", frameon=False)
    plt.title("Density")
    plt.xlabel("False Positive Rate")
    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)
    plt.subplots_adjust(top=0.9)
    plt.savefig("/Users/lls/Documents/CODE/pioneer50/predictions/ROCs_density_3_sims.png")
    plt.clf()


    plt.plot(fpr_shear_den, tpr_shear_den, color="#7ea6ce", label="Planck", lw=2)
    plt.plot(fpr_dsOr, tpr_dsOr, ls="--", color="#7ea6ce", lw=2, label="orig")
    plt.plot(fpr_shear_den_blind, tpr_shear_den_blind, ls="-.", color="#7ea6ce", lw=2, label="blind wmap")
    plt.legend(loc="best", frameon=False)
    plt.title("Density+Shear")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)
    plt.subplots_adjust(top=0.9)
    plt.savefig("/Users/lls/Documents/CODE/pioneer50/predictions/ROCs_shear_density_3_sims.png")