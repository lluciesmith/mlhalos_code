"""
Train and classify only r/r_vir<0.3 particles in halos - test whether there
is an improvement over all particles in the simulation.
Construct training set s.t. you have 2000 particles in each halo mass bin
and 5000 particles in the very last halo bin.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from regression.adaboost import gbm_04_only as gbm_fun


# radii_path = "/home/lls/stored_files/radii_stuff/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
halo_mass_in_ids = halo_mass[halo_mass > 0]

# # sort ids in halos and corresponding r/r_vir value
#
# radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
# radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")
# fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
# ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
# ind_sorted = np.argsort(ids_in_halo)
#
# ids_in_halo_mass = ids_in_halo[ind_sorted].astype("int")
# assert np.allclose(ids_in_halo_mass, np.where(halo_mass > 0)[0])
# r_fraction = fraction[ind_sorted]
# del fraction
# del ids_in_halo
#
#
# # Select a balanced training set
# # Take particle ids in each halo mass bin
#
# n, log_bins = np.histogram(np.log10(halo_mass_in_ids), bins=50)
# bins = 10**log_bins
#
# training_ind = []
# for i in range(len(bins) - 1):
#     ind_bin = np.where((halo_mass_in_ids >= bins[i]) & (halo_mass_in_ids < bins[i + 1]))[0]
#     ids_in_mass_bin = ids_in_halo_mass[ind_bin]
#
#     if ids_in_mass_bin.size == 0:
#         print("Pass")
#         pass
#
#     else:
#         if i == 49:
#             num_p = 2000
#         else:
#             num_p = 1000
#
#         radii_in_mass_bin = r_fraction[ind_bin]
#
#         ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
#         ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
#                                   replace=False)
#         ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
#                                  replace=False)
#         ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)
#
#         training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
#         training_ind.append(training_ids_in_bin)
#
# training_ind = np.concatenate(training_ind)
#
# remaining_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ind)]
# random_sample = np.random.choice(remaining_ids, 50000, replace=False)
#
# training_ind = np.concatenate((training_ind, random_sample))
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_set_2/training_ids.npy",
#         training_ind)
#
# testing_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ind)]
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_set_2/testing_ids.npy",
#         testing_ids)
# y_test = np.log10(halo_mass[testing_ids])
# np.save("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_set_2/log_true_halo_mass"
#         ".npy", y_test)


saving_path = "/share/data2/lls/regression/gradboost/ic_traj/diff_training_set/n_est600/"

# data

training_ind = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_set_2/training_ids.npy")
testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_set_2"
                      "/testing_ids.npy")
y_test = np.log10(halo_mass[testing_ids])

traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")
traj_training = traj[training_ind, :]
log_halo_training = np.log10(halo_mass[training_ind])
training_features = np.column_stack((traj_training, log_halo_training))

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
param_grid = {"loss": "huber",
              "learning_rate": 0.01,
              "n_estimators": 1000,
              "max_depth": 5,
              "max_features": 10,
              "warm_start": True,
              }

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
    ada_r2_test[i] = r2_score(y_test, y_pred)

np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)