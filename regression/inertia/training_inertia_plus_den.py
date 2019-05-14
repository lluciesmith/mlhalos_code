import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# Get training set

#saving_path = "/share/data1/lls/regression/inertia/"
# saving_path = "/share/data1/lls/regression/inertia/min_leaf_1/"

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

# path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
# den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
# den_training = den_features[training_ids]
# den_testing = den_features[testing_ids]
# del den_features
#
# path_inertia = "/share/data1/lls/regression/inertia/cores_40/"
# eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))
# eig_0_training = eig_0[training_ids]
# eig_0_testing = eig_0[testing_ids]
# del eig_0

#### GET FEATURES WITH PERIODICITY FIX FOR DENSITY SMOOTHING ######

path_features = "/share/data2/lls/features_w_periodicity_fix/"
saving_path = "/share/data2/lls/regression/inertia/PERIOD_FIX_inertia_plus_den/"

den_features = np.lib.format.open_memmap(path_features + "ics_density_contrasts.npy", mode="r", shape=(256**3, 50))
den_training = den_features[training_ids]
den_testing = den_features[testing_ids]
del den_features

eig_0 = np.lib.format.open_memmap(path_features + "inertia/eigenvalues_0.npy", mode="r", shape=(256**3, 50))
eig_0_training = eig_0[training_ids]
eig_0_testing = eig_0[testing_ids]
del eig_0

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[training_ids])
del halo_mass


# train

training_features = np.column_stack((den_training, eig_0_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] - 1)/3)
half_features = int((training_features.shape[1] - 1)/2)
quarter_features = int((training_features.shape[1] - 1)/4)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, quarter_features, half_features],
              "min_samples_split": [2, 5],
              "min_samples_leaf": [1]
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "f_imp.npy", clf.feature_importances)

# test

X_test = np.column_stack((den_testing, eig_0_testing))

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", y_predicted)
