import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib

# Get testing set

saving_path = "/share/data1/lls/regression/inertia/"
path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
path_inertia = "/share/data1/lls/regression/inertia/cores_40/"

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
den_testing = den_features[testing_ids]
del den_features

eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))
eig_0_testing = eig_0[testing_ids]
del eig_0

eig_1 = np.lib.format.open_memmap(path_inertia + "eigenvalues_1.npy", mode="r", shape=(256**3, 50))
eig_1_testing = eig_1[testing_ids]
del eig_1

eig_2 = np.lib.format.open_memmap(path_inertia + "eigenvalues_2.npy", mode="r", shape=(256**3, 50))
eig_2_testing = eig_2[testing_ids]
del eig_2

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[testing_ids])
np.save(saving_path + "true_halo_mass.npy", log_mass)
del halo_mass
del log_mass

# test

clf = joblib.load(saving_path + "classifier/classifier.pkl")
X_test = np.column_stack((den_testing, eig_0_testing, eig_1_testing, eig_2_testing))

y_predicted = clf.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)
