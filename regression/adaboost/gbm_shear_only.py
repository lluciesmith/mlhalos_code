import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from regression.adaboost import gbm_04_only as gbm_fun
from sklearn.externals import joblib

# Get training set

saving_path = "/share/data2/lls/regression/gradboost/shear/shear_only/"
features_path = "/share/data2/lls/features_w_periodicity_fix/"

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

den_sub_ell = np.lib.format.open_memmap(features_path + "density_subtracted_ellipticity.npy", mode="r",
                                        shape=(256**3, 50))
ell_training = den_sub_ell[training_ids]
ell_testing = den_sub_ell[testing_ids]
del den_sub_ell

den_sub_prol = np.lib.format.open_memmap(features_path + "density_subtracted_prolateness.npy", mode="r",
                                         shape=(256**3, 50))
prol_training = den_sub_prol[training_ids]
prol_testing = den_sub_prol[testing_ids]
del den_sub_prol

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass_training = np.log10(halo_mass[training_ids])


# train

training_features = np.column_stack((ell_training, prol_training, log_mass_training))
X_test = np.column_stack((ell_testing, prol_testing))
np.save(saving_path + "true_log_mass_test_set.npy", np.log10(halo_mass[testing_ids]))
print(training_features.shape)

cv_i = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"loss": ["lad"],
              "learning_rate": [0.01, 0.05, 0.1],
              "n_estimators": [800, 1000],
              "max_depth": [5],
              "max_features": [third_features]
              }

gbm_CV, pred_test = gbm_fun.train_and_test_gradboost(training_features, X_test, param_grid=param_grid, cv=cv_i)
np.save(saving_path + "predicted_test_set.npy", pred_test)

joblib.dump(gbm_CV, saving_path + "clf.pkl")
np.save(saving_path + "f_imp.npy", gbm_CV.best_estimator_.feature_importances)

