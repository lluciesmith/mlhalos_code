import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# Get training set

saving_path = "/share/data1/lls/reseed50/"
path_simulation = "/home/app/reseed/"

halo_mass_particles = np.load(saving_path + "halo_mass_particles.npy")
ids_in_halo_mass = np.where(halo_mass_particles > 0)[0]

training_ids = np.load("/share/data1/lls/reseed50/regression/100k_training_set.npy")
testing_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ids)]
np.save(saving_path + "regression/testing_ids_100k.npy", testing_ids)
np.save(saving_path + "regression/true_halo_mass_100k_test_set.npy", halo_mass_particles[testing_ids])

den_features = np.lib.format.open_memmap(saving_path + "features/density_contrasts.npy", mode="r", shape=(256**3, 50))
den_training = den_features[training_ids]
den_testing = den_features[testing_ids]
del den_features

eig_0 = np.lib.format.open_memmap(saving_path + "features/inertia/eigenvalues_0.npy", mode="r", shape=(256**3, 50))
eig_0_training = eig_0[training_ids]
eig_0_testing = eig_0[testing_ids]
del eig_0

log_mass_training = np.log10(halo_mass_particles[training_ids])
del halo_mass_particles


# train

training_features = np.column_stack((den_training, eig_0_training, log_mass_training))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, "sqrt", 25, 40],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "regression/den_plus_inertia/classifier/classifier.pkl",
                     param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "regression/den_plus_inertia/f_imp.npy", clf.feature_importances)

# test

X_test = np.column_stack((den_testing, eig_0_testing))

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "regression/den_plus_inertia/predicted_halo_mass.npy", 10**y_predicted)
