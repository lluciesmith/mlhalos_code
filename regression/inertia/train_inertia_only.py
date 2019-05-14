import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from sklearn.preprocessing import RobustScaler

# Get training set
path_inertia = "/share/data1/lls/regression/inertia/cores_40/"

saving_path = "/share/data2/lls/regression/inertia/inertia_only/100k_random_training/rescaling_feat/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
in_halos_ids = np.where(halo_mass > 0)[0]
training_ids = np.random.choice(in_halos_ids, 100000, replace=False)
np.save(saving_path + "training_set.npy", training_ids)

# saving_path = "/share/data2/lls/regression/inertia/inertia_only/"
# training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids
# .npy")

# testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))

eig_0 = RobustScaler().fit_transform(eig_0)
eig_0_training = eig_0[training_ids]
# eig_0_testing = eig_0[testing_ids]
del eig_0

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[training_ids])
del halo_mass


# train

training_features = np.column_stack((eig_0_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [800, 1000, 1300],
              "max_features": [third_features, "sqrt", 5, 10],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "f_imp.npy", clf.feature_importances)

# # test
#
# y_predicted = clf.algorithm.predict(eig_0_testing)
# np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)
