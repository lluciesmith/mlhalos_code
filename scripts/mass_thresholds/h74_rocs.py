import numpy as np
from mlhalos import machinelearning as ml
from mlhalos import parameters
from scripts.EPS import EPS_predictions as EPS_pred
from scripts.ellipsoidal import predictions as ST_pred
import matplotlib.pyplot as plt

# ROCs

path = "/Users/lls/Documents/mlhalos_files/stored_files/h74_in_out_boundary/"

pred_density = np.load(path + "density/predicted_probabilities.npy")
true_density = np.load(path + "density/true_test_labels.npy")
fpr_den, tpr_den, auc_den, threshold_den = ml.roc(pred_density, true_density)

pred_shear = np.load(path + "shear_density/predicted_probabilities.npy")
true_shear = np.load(path + "shear_density/true_test_labels.npy")
fpr_shear, tpr_shear, auc_shear, threshold_shear = ml.roc(pred_shear, true_shear, true_class=1)

# EPS and ST

try:
    EPS_predicted_label = np.load(path + "EPS_predicted_label.npy")
    ST_predicted_label = np.load(path + "ST_predicted_label.npy")
except IOError:
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files",
                                                load_final=True, max_halo_number=74)
    density_traj = np.load("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/density_trajectories.npy")

    EPS_predicted_label = EPS_pred.EPS_label(density_traj, initial_parameters=ic)
    np.save(path + "EPS_predicted_label.npy", EPS_predicted_label)

    ST_predicted_label = ST_pred.ellipsoidal_collapse_predicted_label(density_traj, initial_parameters=ic,
                                                                      beta=0.485, gamma=0.615, a=0.707)
    np.save(path + "ST_predicted_label.npy", ST_predicted_label)

ic = parameters.InitialConditionsParameters(load_final=True, max_halo_number=74)
num_ids = len(ic.final_snapshot)

true_labels = np.ones((num_ids,))
true_labels[ic.ids_OUT] *= -1

fpr_EPS, tpr_EPS = ST_pred.get_fpr_tpr_ellipsoidal_prediction(EPS_predicted_label, true_labels)
fpr_ST, tpr_ST = ST_pred.get_fpr_tpr_ellipsoidal_prediction(ST_predicted_label, true_labels)

# plot

plt.plot(fpr_den, tpr_den, color="#8856a7", lw=1.5, label="Density (AUC = " + ' %.3f' % (auc_den) + ")")
plt.plot(fpr_shear, tpr_shear, color="#7ea6ce", lw=1.5, label="Density+Shear (AUC = " + ' %.3f' % (auc_shear) + ")")

plt.plot(fpr_EPS, tpr_EPS, "ko", label="EPS prediction")
plt.plot(fpr_ST, tpr_ST, "k^", label="ST prediction")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(-0.03, 1.03)
plt.ylim(-0.03, 1.03)
plt.legend(loc="best", frameon=False)
plt.subplots_adjust(top=0.9)
plt.title(r"$M_{\mathrm{IN/OUT}} = 10^{13} M_\odot$")
plt.savefig(path + "rocs.pdf")