"""
Train only IN HALOS particles

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

##### TRAINING SET #######

# constructed balanced training set - equal number of particles in each bin

# saving_path = "/share/data1/lls/regression/in_halos_only/balanced_training_set/"

# halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
# halos_in = halo_mass[halo_mass > 0]
# ids_in = np.where(halo_mass > 0)[0]
#
# n, log_bins = np.histogram(np.log10(halos_in), bins=50)
# bins = 10**log_bins
#
# training_ind = []
# for i in range(len(bins) - 1):
#     p_in_bins = ids_in[(halos_in >= bins[i]) & (halos_in < bins[i + 1])]
#     if p_in_bins.size == 0:
#         print("Pass")
#         pass
#     else:
#         ind = np.random.choice(p_in_bins, 2000, replace=False)
#         training_ind.append(ind)
#
# training_ind = np.concatenate(training_ind)
# np.save(saving_path + "training_ids.npy", training_ind)
#
# testing_ind = ids_in[~np.in1d(ids_in, training_ind)]
# np.save(saving_path + "testing_ids.npy", testing_ind)
# y_test = halo_mass[testing_ind]
# np.save(saving_path + "true_halo_mass.npy", y_test)

# Random subset

saving_path = "/share/data1/lls/regression/in_halos_only/log_m_output/CV_extended/"

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

in_halos_ids = np.where(halo_mass > 0)[0]
training_ind = np.random.choice(in_halos_ids, 100000, replace=False)
np.save(saving_path + "training_set.npy", training_ind)

testing_ind = in_halos_ids[~np.in1d(in_halos_ids, training_ind)]
np.save(saving_path + "testing_ind.npy", testing_ind)
y_test = halo_mass[testing_ind]
np.save(saving_path + "true_halo_mass.npy", y_test)


##### TRAINING #######

traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")
log_mass = np.log10(halo_mass)

feat_training = np.column_stack((traj[training_ind], log_mass[training_ind]))
X_test = traj[testing_ind]
del traj
del halo_mass

cv = True
clf = ml.MLAlgorithm(feat_training, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60, save=True,
                     path=saving_path + "classifier/classifier.pkl")
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)
np.save(saving_path + "f_imp.npy", clf.feature_importances)

# classify

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)