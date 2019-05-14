import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# Get training set

saving_path = "/share/data1/lls/regression/shear/shear_only/"
path_features = "/share/data1/lls/shear_quantities/quantities_id_ordered/"

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

den_sub_ell = np.lib.format.open_memmap(path_features + "density_subtracted_ellipticity.npy", mode="r",
                                        shape=(256**3, 50))
ell_training = den_sub_ell[training_ids]
ell_testing = den_sub_ell[testing_ids]
del den_sub_ell

den_sub_prol = np.lib.format.open_memmap(path_features + "density_subtracted_prolateness.npy", mode="r",
                                         shape=(256**3, 50))
prol_training = den_sub_prol[training_ids]
prol_testing = den_sub_prol[testing_ids]
del den_sub_prol

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[training_ids])
del halo_mass


# train

training_features = np.column_stack((ell_training, prol_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [800, 1000, 1300],
              "max_features": [third_features, "sqrt", 20, 30],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60, save=True,
                     path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)
np.save(saving_path + "f_imp.npy", clf.feature_importances)

#test
X_test = np.column_stack((ell_testing, prol_testing))

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)
