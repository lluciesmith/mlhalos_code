import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 18})
from mlhalos import plot
from mlhalos import machinelearning as ml
from mlhalos import distinct_colours
from utils import mass
import numpy as np
from mlhalos import parameters
import matplotlib.pyplot as plt
import matplotlib
import importlib
importlib.reload(mass)
from scripts.ellipsoidal import predictions as ell_pred


############ ROC PLOT #############


def get_roc_plot(y_proba, y_true, labels=[""], add_EPS=False, fpr_EPS=None, tpr_EPS=None, add_ellipsoidal=False,
                 fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_EPS=None, label_ellipsoidal=None,
                 fontsize_labels=17, color="#8856a7"):

    fpr, tpr, auc, threshold = ml.roc(y_proba, y_true, true_class=1)
    fig = plot.roc_plot(fpr, tpr, auc, labels=labels, figsize=None, add_EPS=add_EPS, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS,
                  add_ellipsoidal=add_ellipsoidal, fpr_ellipsoidal=fpr_ellipsoidal, tpr_ellipsoidal=tpr_ellipsoidal,
                  label_EPS=label_EPS, label_ellipsoidal=label_ellipsoidal, frameon=False,
                        fontsize_labels=fontsize_labels, cols=color)
    return fig


def get_EPS_labels(EPS_IN_labels, EPS_OUT_labels):
    tpr = len(np.where(EPS_IN_labels == 1)[0]) / len(EPS_IN_labels)
    fpr = len(np.where(EPS_OUT_labels == 1)[0]) / len(EPS_OUT_labels)
    return fpr, tpr


def get_multiple_rocs(predicted, true, labels=[""], add_EPS=False, fpr_EPS=None, tpr_EPS=None, add_ellipsoidal=False,
                      fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_EPS=None, label_ellipsoidal=None,
                      fontsize_labels=17):
    FPR = np.zeros((2, 50))
    TPR = np.zeros((2, 50))
    AUC = np.zeros((2, ))
    for i in range(2):
        FPR[i], TPR[i], AUC[i], threshold = ml.roc(predicted[i], true[i], true_class=1)

    fig = plot.roc_plot(FPR.transpose(), TPR.transpose(), AUC, labels=labels, figsize=None, add_EPS=add_EPS,
                        fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS, add_ellipsoidal=add_ellipsoidal,
                        fpr_ellipsoidal=fpr_ellipsoidal, tpr_ellipsoidal=tpr_ellipsoidal, label_EPS=label_EPS,
                        label_ellipsoidal=label_ellipsoidal, frameon=False,
                        fontsize_labels=fontsize_labels, cols=["#8856a7", "#7ea6ce" ])

    return fig


if __name__ == "__main__":
    #figure_number = sys.argv[1]
    # print(figure_number)
    # save = sys.argv[2]

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

    ####################### FIGURE ROC DENSITY  #######################

    pred_density = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/"
                           "predicted_den.npy")
    true_density = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/true_den.npy")
    fpr, tpr, auc, threshold = ml.roc(pred_density, true_density, true_class=1)

    EPS_IN_label = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_IN.npy")
    EPS_OUT_label = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_OUT.npy")
    fpr_EPS, tpr_EPS = get_EPS_labels(EPS_IN_label, EPS_OUT_label)

    # fig = get_roc_plot(pred_density, true_density, add_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS, labels=["Density"]
    #                    # label_EPS="EPS"
    #                    )
    # plt.savefig("/Users/lls/Documents/talks/density_only_ROC.pdf")
    # # fig.text(fpr_EPS + 0.15, tpr_EPS,'EPS', verticalalignment='center',
    # #          size=matplotlib.rcParams['font.size'])
    #
    # plt.savefig("/Users/lls/Documents/mlhalos_paper/Figure_1.pdf")


    ####################### FIGURE ROC DENSITY + SHEAR #######################
    den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")
    fpr_ST, tpr_ST = ell_pred.get_fpr_tpr_from_features(den_f, mass_range="in", initial_parameters=None,
                                                        window_parameters=None, beta=0.485, gamma=0.615, a=0.707)

    path = "/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol/"

    pred_shear = np.load(path + "predicted_den+den_sub_ell+den_sub_prol.npy")
    true_shear = np.load(path + "true_den+den_sub_ell+den_sub_prol.npy")

    pred_all = np.array([pred_density, pred_shear])
    true_all = np.array([true_density, true_shear])
    np.testing.assert_allclose(true_density, true_shear)

    f_shear = get_multiple_rocs(pred_all, true_all, labels=["Density", "Density+Shear"],
                                add_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS,
                                add_ellipsoidal=True, fpr_ellipsoidal=fpr_ST, tpr_ellipsoidal=tpr_ST)

    plt.savefig("/Users/lls/Documents/mlhalos_paper/ROC_density_shear_new.pdf")

    # def ind_near(array, value):
    #     return (np.abs(array - value)).argmin()
    #
    # print("EPS TPR are FPR are " + str(tpr_EPS) + " and " + str(fpr_EPS))
    # print("ST TPR are FPR are " + str(tpr_ST) + " and " + str(fpr_ST))
    # print(tpr)
    # print(fpr)
    #
    # print("The nearest index between density TPR and TPR EPS is " + str(ind_near(tpr, tpr_EPS)))
    # print("The nearest index between density FPR and FPR EPS is " + str(ind_near(fpr, fpr_EPS)))
    # print("The nearest index between density TPR and TPR ST is " + str(ind_near(tpr, tpr_ST)))
    # print("The nearest index between density fPR and FPR ST is " + str(ind_near(fpr, fpr_ST)))




# This doesn't really work but didnt want to delete it! Use `mass_radius_bins_rocs.py` to make this plot though

############ MASS BINS ROC PLOT #############


def get_particle_ids_test_set(num_particles, training_indices):
    all_ids = np.arange(num_particles)
    testing_ids = all_ids[~np.in1d(np.arange(num_particles), training_indices)]
    return testing_ids


def split_predicted_and_true_labels_three_categories(num_particles, training_indices, predicted, true,
                                                     cat_1, cat_2, cat_3):
    testing_ids = get_particle_ids_test_set(num_particles, training_indices)
    Y_PROBA = [predicted[np.in1d(testing_ids, cat_1)],
               predicted[np.in1d(testing_ids, cat_2)],
               predicted[np.in1d(testing_ids, cat_3)]]
    Y_TRUE = [true[np.in1d(testing_ids, cat_1)],
              true[np.in1d(testing_ids, cat_2)],
              true[np.in1d(testing_ids, cat_3)]]
    return Y_PROBA, Y_TRUE


def plot_rocs_3_categories(Y_PROBA, Y_TRUE, labels=[""]):
    FPR = np.zeros((3, 50))
    TPR = np.zeros((3, 50))
    AUC = np.zeros((3, ))
    for i in range(3):
        FPR[i], TPR[i], AUC[i], threshold = ml.roc(Y_PROBA[i], Y_TRUE[i], true_class=1)

    plot.roc_plot(FPR, TPR, AUC, labels=labels, figsize=None, add_EPS=False, fpr_EPS=None, tpr_EPS=None,
                  add_ellipsoidal=False, fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_EPS=None,
                  label_ellipsoidal=None, frameon=False)


def pred_split_for_mass_bins(y_proba, y_true, training_indices, initial_parameters, high_halo=6, mid_halo=78):
    num_particles = len(initial_parameters.initial_conditions)

    # Split predicted and true in 3 mass bins

    high_mass_particle_ids = mass.get_particles_in_mass_bin(mass_bin="high", high_halo=high_halo, mid_halo=mid_halo,
                                                            initial_parameters=initial_parameters)
    mid_mass_particle_ids = mass.get_particles_in_mass_bin(mass_bin="mid", high_halo=high_halo, mid_halo=mid_halo,
                                                           initial_parameters=initial_parameters)
    low_mass_particle_ids = mass.get_particles_in_mass_bin(mass_bin="small", high_halo=high_halo, mid_halo=mid_halo,
                                                           initial_parameters=initial_parameters)

    Y_PROBA, Y_TRUE = split_predicted_and_true_labels_three_categories(num_particles, training_indices, y_proba, y_true,
                                                                       high_mass_particle_ids, mid_mass_particle_ids,
                                                                       low_mass_particle_ids)
    return Y_PROBA, Y_TRUE


def get_roc_plot_mass_bins(y_proba, y_true, training_indices, initial_parameters, high_halo=6, mid_halo=78):
    """
    - The ``high" mass bins is all halos of mass M >= mass of `high halo'
    - The ``mid" mass bins is all halos of mass (mass of `high halo') < M <= (mass of 'mid halo')
    - The ``high" mass bins is all halos of mass M > mass of `high halo' M < mass of `mid halo'

    """

    Y_PROBA, Y_TRUE = pred_split_for_mass_bins(y_proba, y_true, training_indices, initial_parameters,
                                               high_halo=high_halo, mid_halo=mid_halo)

    # Get FPR and TPR for threee mass bins separately

    high_mass = ic.halo[high_halo]['mass'].sum()
    mid_mass = ic.halo[mid_halo]['mass'].sum()

    labels = [r"$\mathrm{M} > $" + " %.3e " % high_mass + r"$\mathrm{M}_{\odot}$",
              " %.3e" % high_mass + r"$ < \mathrm{M}/\mathrm{M}_{\odot} > $" + " %.3e" % mid_mass,
              r"$\mathrm{M} < $" + " %.3e " % mid_mass + r"$\mathrm{M}_{\odot}$"]
    plot_rocs_3_categories(Y_PROBA, Y_TRUE, labels=labels)





