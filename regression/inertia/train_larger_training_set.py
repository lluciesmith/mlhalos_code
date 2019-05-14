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
saving_path = "/share/data1/lls/regression/inertia/large_training_set/"

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
            # num_p = 2000
            num_p = 4000
        else:
            # num_p = 1000
            num_p = 2500

        radii_in_mass_bin = r_fraction[ind_bin]

        ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
        print((num_p/len(ids_in_mass_bin[radii_in_mass_bin < 0.3]) * 100))
        ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
                                  replace=False)
        print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)]) * 100))
        ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
                                 replace=False)
        print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)]) * 100))
        ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)
        print((num_p / len(ids_in_mass_bin[radii_in_mass_bin >= 1]) * 100))

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

path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
den_training = den_features[training_ind]
den_testing = den_features[testing_ids]
del den_features

path_inertia = "/share/data1/lls/regression/inertia/cores_40/"
eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))
eig_0_training = eig_0[training_ind]
eig_0_testing = eig_0[testing_ids]
del eig_0

log_mass_training = np.log10(halo_mass[training_ind])

feat_training = np.column_stack((den_training, eig_0_training, log_mass_training))

cv = True
third_features = int((feat_training.shape[1] - 1)/3)
half_features = int((feat_training.shape[1] - 1)/2)
quarter_features = int((feat_training.shape[1] - 1)/4)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, quarter_features, half_features],
              "min_samples_split": [2, 5],
              "min_samples_leaf": [1]
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(feat_training, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)
np.save(saving_path + "f_imp.npy", clf.feature_importances)

# classify

X_test = np.column_stack((den_testing, eig_0_testing))
y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)
