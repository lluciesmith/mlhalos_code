"""
Train and classify only particles s.t.:
- ONLY PARTICLES IN HALOS
- LOG M OUTPUT

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml


saving_path = "/share/data1/lls/regression/in_halos_only/log_m_output/balanced_training_set/"

# Select a balanced training set

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
all_ids = np.arange(256**3)
traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")

n, log_bins = np.histogram(np.log10(halo_mass[halo_mass>0]), bins=50)
bins = 10**log_bins

training_ind = []
for i in range(len(bins) - 1):
    p_in_bins = all_ids[(halo_mass >= bins[i]) & (halo_mass < bins[i + 1])]
    if p_in_bins.size == 0:
        print("Pass")
        pass
    elif i == 49:
        ind = np.random.choice(p_in_bins, 5000, replace=False)
        training_ind.append(ind)
    else:
        ind = np.random.choice(p_in_bins, 2000, replace=False)
        training_ind.append(ind)

training_ind = np.concatenate(training_ind)

np.save(saving_path + "training_ids.npy", training_ind)

in_halos_ids = np.where(halo_mass > 0)[0]
testing_indices = ~np.in1d(in_halos_ids, training_ind)
testing_ids = in_halos_ids[testing_indices]
np.save(saving_path + "testing_ids.npy", testing_ids)
y_test = halo_mass[testing_ids]
np.save(saving_path + "true_halo_mass.npy", y_test)

##### TRAINING + PREDICTING #######

log_mass = np.log10(halo_mass)

feat_training = np.column_stack((traj[training_ind], log_mass[training_ind]))
X_test = traj[testing_ids]

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
