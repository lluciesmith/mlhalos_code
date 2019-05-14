import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

saving_path = "/share/data2/lls/regression/local_inertia/no_cv/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

training_ids = np.load("/share/data2/lls/regression/local_inertia/tensor/first_try/training_particles_saved.npy")
reduced_ids = np.load("/share/data2/lls/regression/local_inertia/reduced_training_set.npy")

indices_reduced = np.in1d(training_ids, reduced_ids)

eig_training = np.load("/share/data2/lls/regression/local_inertia/tensor/first_try/training_eigenvalues_particles.npy")
eig_reduced = eig_training[indices_reduced]

eig0 = eig_reduced[:, :, 0]
eig1 = eig_reduced[:, :, 1]
eig2 = eig_reduced[:, :, 2]

path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256 ** 3, 50))
den_training = den_features[reduced_ids]

log_mass = np.log10(halo_mass[reduced_ids])

# train

training_features = np.column_stack((den_training, eig0, eig1, eig2, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] - 1) / 3)
param_grid = {"n_estimators": [1600],
              "max_features": [third_features],
              "min_samples_split": [2],
              "min_samples_leaf": [1]
              # "criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "f_imp.npy", clf.feature_importances)
