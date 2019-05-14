"""
Train and classify only r/r_vir<0.3 particles in halos - test whether there
is an improvement over all particles in the simulation.
Construct training set s.t. you have 2000 particles in each halo mass bin
and 5000 particles in the very last halo bin.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

radii_path = "/home/lls/stored_files/radii_stuff/"
saving_path = "/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/"

# Select only particles which are within 30% of virial radius

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
halo_mass_in_ids = halo_mass[halo_mass > 0]

# sort ids in halos and corresponding r/r_vir value

radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")
fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
ind_sorted = np.argsort(ids_in_halo)

ids_in_halo_mass = ids_in_halo[ind_sorted].astype("int")
r_fraction = fraction[ind_sorted]
del fraction
del ids_in_halo


# Select a balanced training set
# Take particle ids in each halo mass bin

n, log_bins = np.histogram(np.log10(halo_mass_in_ids), bins=50)
bins = 10**log_bins

training_ind = []
for i in range(len(bins) - 1):
    ind_bin = np.where((halo_mass_in_ids >= bins[i]) & (halo_mass_in_ids < bins[i + 1]))[0]
    ids_in_mass_bin = ids_in_halo_mass[ind_bin]

    if ids_in_mass_bin.size == 0:
        print("Pass")
        pass

    else:
        if i == 49:
            num_p = 2000
        else:
            num_p = 1000

        radii_in_mass_bin = r_fraction[ind_bin]

        ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
        ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
                                  replace=False)
        ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
                                 replace=False)
        ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)

        training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
        training_ind.append(training_ids_in_bin)

training_ind = np.concatenate(training_ind)

remaining_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ind)]
random_sample = np.random.choice(remaining_ids, 50000, replace=False)

training_ind = np.concatenate((training_ind, random_sample))
np.save(saving_path + "training_ids.npy", training_ind)

testing_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ind)]
np.save(saving_path + "testing_ids.npy", testing_ids)
y_test = halo_mass[testing_ids]
np.save(saving_path + "true_halo_mass.npy", y_test)


##### TRAINING + PREDICTING #######

traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")
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
