"""
Predict halo mass of test set using trained regression classifier

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib

traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
testing_ind = np.load("/share/data1/lls/regression/50k_testing_ids.npy")

X_test = traj[testing_ind]
y_test = halo_mass[testing_ind]
del traj
del halo_mass

algo = joblib.load("/share/data1/lls/regression/classifier/classifier.pkl")

y_predicted = algo.predict(X_test)
np.save("/share/data1/lls/regression/predicted_halo_mass.npy", y_predicted)
np.save("/share/data1/lls/regression/true_halo_mass.npy", y_test)
