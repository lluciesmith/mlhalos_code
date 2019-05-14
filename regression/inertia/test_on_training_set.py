import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib

saving_path = "/share/data1/lls/regression/inertia/predict_training/"

clf = joblib.load("/share/data1/lls/regression/inertia/classifier/classifier.pkl")

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")


path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
den_training = den_features[training_ids]
del den_features

path_inertia = "/share/data1/lls/regression/inertia/cores_40/"
eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))
eig_0_training = eig_0[training_ids]
del eig_0

X_test = np.column_stack((den_training, eig_0_training))

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_halo_mass_training_set.npy", 10**y_predicted)
