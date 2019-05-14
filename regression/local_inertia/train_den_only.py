import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from regression.local_inertia import train_RF as t_rf

# Get training set

saving_path = "/share/data2/lls/regression/local_inertia/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

# training_ids = np.load("/share/data2/lls/regression/local_inertia/tensor/first_try/training_particles_saved.npy")
reduced_ids = np.load(saving_path + "reduced_training_set.npy")

path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
den_training = den_features[reduced_ids]

log_mass = np.log10(halo_mass[reduced_ids])

# train density only

training_features = np.column_stack((den_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] - 1)/3)
half_features = int((training_features.shape[1] - 1)/2)
quarter_features = int((training_features.shape[1] - 1)/4)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, quarter_features, half_features],
              "min_samples_split": [2],
              "min_samples_leaf": [1]
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "clf_den_only/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "clf_den_only/f_imp.npy", clf.feature_importances)
