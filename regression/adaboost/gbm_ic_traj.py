"""
Test what happens when you increase the number of training examples.
In particular, in increasing the number of particles at the edges of the distribution.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from mlhalos import machinelearning as ml
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from regression.adaboost import gbm_04_only as gbm_fun

saving_path = "/share/data2/lls/regression/gradboost/ic_traj/loss_lad_lr001_n1000/"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
traj_training = traj[training_ids, :]
log_halo_training = np.log10(halo_mass[training_ids])
training_features = np.column_stack((traj_training, log_halo_training))

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
log_halo_testing = np.log10(halo_mass[testing_ids])
traj_testing = traj[testing_ids, :]

del traj
del halo_mass
#
# cv_i = True
# param_grid = {"loss": ["huber"],
#               "learning_rate": [0.001, 0.01, 0.05],
#               "n_estimators": [1000, 1500],
#               "max_depth": [5, 10],
#               "max_features": [0.3, "sqrt"]
#               }
cv_i = False
# param_grid = {"loss": "huber",
#               "learning_rate": 0.01,
#               "n_estimators": 300,
#               "max_depth": 5,
#               "max_features": "sqrt",
#               "warm_start": True,
#               "subsample": 0.8
#               }
param_grid = {"loss": "lad",  "learning_rate": 0.01,  "n_estimators": 1000, "max_depth": 5, "max_features": 15}

gbm_CV, pred_test = gbm_fun.train_and_test_gradboost(training_features, traj_testing,
                                                     param_grid=param_grid, cv=cv_i)
np.save(saving_path + "predicted_test_set.npy", pred_test)
joblib.dump(gbm_CV, saving_path + "clf.pkl")

# predictions

if cv_i is True:
    alg = gbm_CV.best_estimator_
    ml.write_to_file_cv_results(saving_path + "cv_results.txt", gbm_CV)
else:
    alg = gbm_CV

ada_r2_train = np.zeros(len(alg.estimators_), )
for i, y_pred in enumerate(alg.staged_predict(training_features[:, :-1])):
    ada_r2_train[i] = r2_score(log_halo_training, y_pred)

np.save(saving_path + "r2_train_staged_scores.npy", ada_r2_train)

ada_r2_test = np.zeros(len(alg.estimators_), )
for i, y_pred in enumerate(alg.staged_predict(traj_testing)):
    ada_r2_test[i] = r2_score(log_halo_testing, y_pred)

np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)