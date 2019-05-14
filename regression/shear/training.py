import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# Get training set

saving_path = "/share/data1/lls/regression/shear/density_period_fix/"
path_features = "/share/data2/lls/features_w_periodicity_fix/"

# saving_path = "/share/data1/lls/regression/shear/"
# path_features = "/share/data1/lls/shear_quantities/quantities_id_ordered/"

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")

den_features = np.lib.format.open_memmap(path_features + "ics_density_contrasts.npy", mode="r", shape=(256**3, 50))
# den_features = np.lib.format.open_memmap(path_features + "density_trajectories.npy", mode="r", shape=(256**3, 50))
den_training = den_features[training_ids]
del den_features

den_sub_ell = np.lib.format.open_memmap(path_features + "density_subtracted_ellipticity.npy", mode="r",
                                        shape=(256**3, 50))
ell_training = den_sub_ell[training_ids]
del den_sub_ell

den_sub_prol = np.lib.format.open_memmap(path_features + "density_subtracted_prolateness.npy", mode="r",
                                         shape=(256**3, 50))
prol_training = den_sub_prol[training_ids]
del den_sub_prol

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[training_ids])
del halo_mass


# train

training_features = np.column_stack((den_training, ell_training, prol_training, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [1000, 1300, 1600],
              "max_features": [third_features, "sqrt", 25, 40],
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

