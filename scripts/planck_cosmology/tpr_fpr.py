"""
Planck-like cosmology simulation pioneer50:
ICs - /share/data1/lls/pioneer50/pioneer50.512.ICs.tipsy
Final snapshot - /share/data1/lls/pioneer50/pioneer50.512.004096

"""

import sys
sys.path.append('/home/lls/mlhalos_code')
import numpy as np
from mlhalos import machinelearning as ml


if __name__ == "__main__":
    path = "/share/data1/lls/pioneer50/predictions/"
    true_labels = np.load(path + "true_labels.npy")

    density_pred = np.load(path + "density_predicted_probabilities.npy")
    fpr_den, tpr_den, auc_den, thr = ml.roc(density_pred, true_labels)
    print(auc_den)
    np.save(path + "fpr_den.npy", fpr_den)
    np.save(path + "tpr_den.npy", tpr_den)
    del density_pred
    del fpr_den
    del tpr_den

    density_shear_pred = np.load(path + "shear_density_predicted_probabilities.npy")
    fpr_shear_den, tpr_shear_den, auc_shear_den, thr_shear = ml.roc(density_shear_pred, true_labels)
    print(auc_shear_den)
    np.save(path + "fpr_shear_den.npy", fpr_shear_den)
    np.save(path + "tpr_shear_den.npy", tpr_shear_den)