import numpy as np
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
from scripts.EPS import EPS_predictions as p

EPS_IN_label = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_IN.npy")
EPS_OUT_label = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predictions_OUT.npy")
EPS_probabilities_all = np.concatenate((EPS_IN_label, EPS_OUT_label))

den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")
training_index = np.load('/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy')
den_f_test = den_f[~np.in1d(np.arange(len(den_f)), training_index)]
pred = p.EPS_label(den_f_test[:, :-1], mass_range="in", initial_parameters=None)
np.save("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/EPS_predicted_labels.npy", pred)

print(np.allclose(EPS_probabilities_all, pred))