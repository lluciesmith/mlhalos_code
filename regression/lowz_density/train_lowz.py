import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from sklearn.externals import joblib

# Get training set

features_path = "/share/data2/lls/features_w_periodicity_fix/"
saving_path = "/share/data2/lls/regression/lowz_density/"

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

ics_den_features = np.lib.format.open_memmap(features_path + "ics_density_contrasts.npy", mode="r", shape=(256**3, 50))
ics_den_training = ics_den_features[training_ids]
ics_den_testing = ics_den_features[testing_ids]
del ics_den_features

# z8_den_features = np.lib.format.open_memmap(saving_path + "z8_density_contrasts.npy", mode="r", shape=(256**3, 50))
# z8_den_training = z8_den_features[training_ids]
# z8_den_testing = z8_den_features[testing_ids]
# del z8_den_features

# z01_den_features = np.lib.format.open_memmap(features_path + "z01_density_contrasts.npy", mode="r", shape=(256**3, 50))
# z01_den_training = z01_den_features[training_ids]
# z01_den_testing = z01_den_features[testing_ids]
# del z01_den_features

z0_den_features = np.lib.format.open_memmap(features_path + "z0l_density_contrasts.npy", mode="r", shape=(256**3, 50))
z0_den_training = z0_den_features[training_ids]
z0_den_testing = z0_den_features[testing_ids]
del z0_den_features

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[training_ids])
# log_mass_test_set = np.log10(halo_mass[testing_ids])
# np.save(saving_path + "true_mass_test_set.npy", log_mass_test_set)
del halo_mass


# train on ICS + LOW-Z

# training_features = np.column_stack((ics_den_training, z8_den_training, log_mass))
training_features = np.column_stack((ics_den_training, z0_den_training, log_mass))
# training_features = np.column_stack((ics_den_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, "sqrt", 25, 40],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "z0/ics_plus_z0/classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "z0/ics_plus_z0/f_imp.npy", clf.feature_importances)

# test
X_test = np.column_stack((ics_den_testing, z0_den_testing))
# X_test = ics_den_testing
# X_test = np.column_stack((ics_den_testing, z8_den_testing))
# y_predicted = clf.algorithm.predict(X_test)


y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "z0/ics_plus_z0/predicted_log_halo_mass.npy", y_predicted)
del clf
del y_predicted


# train on LOW-Z only

# training_features = np.column_stack((ics_den_training, z8_den_training, log_mass))
training_features = np.column_stack((z0_den_training, log_mass))
# training_features = np.column_stack((ics_den_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, "sqrt", 25, 40],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "z0/z0_only/classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "z0/z0_only/f_imp.npy", clf.feature_importances)

# test
X_test = z0_den_testing
# X_test = np.column_stack((ics_den_testing, z01_den_testing))
# X_test = ics_den_testing
# X_test = np.column_stack((ics_den_testing, z8_den_testing))
# y_predicted = clf.algorithm.predict(X_test)

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "z0/z0_only/predicted_log_halo_mass.npy", y_predicted)