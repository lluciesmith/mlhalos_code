"""
Construct training set s.t. you have 1000 particles in each halo mass bin
and 2000 particles in no halos.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# constructed balanced training set - save training/testing ids

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
n, log_bins = np.histogram(np.log10(halo_mass[halo_mass > 0]), bins=50)
bins = 10**log_bins

indices = np.arange(len(halo_mass))
training_ind = []
for i in range(len(bins) - 1):
    p_in_bins = indices[(halo_mass >= bins[i]) & (halo_mass < bins[i + 1])]
    if p_in_bins.size == 0:
        print("Pass")
        pass
    else:
        ind = np.random.choice(p_in_bins, 1000, replace=False)
        training_ind.append(ind)

ind_no_halo = indices[halo_mass == 0]
training_ind.append(np.random.choice(ind_no_halo, 2000, replace=False))

training_ind = np.concatenate(training_ind)
np.save("/share/data1/lls/regression/balanced_training_set/training_ids.npy", training_ind)

testing_ind = indices[~np.in1d(indices, training_ind)]
np.save("/share/data1/lls/regression/balanced_training_set/testing_ids.npy", testing_ind)
y_test = halo_mass[testing_ind]
np.save("/share/data1/lls/regression/balanced_training_set/true_halo_mass.npy", y_test)

# train

traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")

feat_training = np.column_stack((traj[training_ind], halo_mass[training_ind]))
X_test = traj[testing_ind]
del traj
del halo_mass

cv = False
clf = ml.MLAlgorithm(feat_training, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True,
                     path="/share/data1/lls/regression/balanced_training_set/classifier/classifier.pkl")

if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)
np.save("/share/data1/lls/regression/balanced_training_set/f_imp.npy", clf.feature_importances)

# classify

y_predicted = clf.algorithm.predict(X_test)
np.save("/share/data1/lls/regression/balanced_training_set/predicted_halo_mass.npy", y_predicted)

