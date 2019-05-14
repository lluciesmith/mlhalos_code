
"""
Test gradboost on z=0 density features

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

saving_path = "/share/data2/lls/regression/gradboost/z0_den/"
features_path = "/share/data2/lls/features_w_periodicity_fix/"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

z0_den_features = np.load(features_path + "z0l_density_contrasts.npy")

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
z0_den_training = z0_den_features[training_ids]
log_halo_training = np.log10(halo_mass[training_ids])
training_features = np.column_stack((z0_den_training, log_halo_training))

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
log_halo_testing = np.log10(halo_mass[testing_ids])
z0_den_testing = z0_den_features[testing_ids]


del halo_mass
#
# cv_i = True
# param_grid = {"loss": ["huber"],
#               "learning_rate": [0.001, 0.01, 0.05],
#               "n_estimators": [1000, 1500],
#               "max_depth": [5, 10],
#               "max_features": [0.3, "sqrt"]
#               }
# cv_i = False
# param_grid = {"loss": "huber",
#               "learning_rate": 0.03,
#               "n_estimators": 1500,
#               "max_depth": 5,
#               "max_features": 20
#               }
cv_i = False
param_grid = {"loss": "lad",
              "learning_rate": 0.01,
              "n_estimators": 1000,
              "max_depth": 5,
              "max_features": 15,
              #"alpha": 0.5
              }

gbm_CV, pred_test = gbm_fun.train_and_test_gradboost(training_features, z0_den_testing, param_grid=param_grid, cv=cv_i)
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
for i, y_pred in enumerate(alg.staged_predict(z0_den_testing)):
    ada_r2_test[i] = r2_score(log_halo_testing, y_pred)

np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)




##### Why are particles being misclassified

# Take particles in smallest mass bin which are being misclassified and which are not being misclassified.
# Compare their features histograms

# pred = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/z0_den/predicted_test_set.npy")
# true = np.log10(np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random"
#                         "/true_halo_mass.npy"))
#
# ids_smallest_correct = np.where((true <= 11) & (pred <= 11))[0]
# ids_smallest_incorrect = np.where((true <= 11) & (pred >= 12.5)& (pred <= 12.8))[0]
# ids_correct_pred_mid = np.where((true >12.5) & (pred >= 12.5) & (true <12.8) & (pred <= 12.8))[0]
#
# for i in range(50):
#     plt.figure()
#     plt.hist(z0_testing[ids_correct_pred_mid, i], bins=50, label="correct(mid mass)", histtype="step", normed=True)
#     plt.hist(z0_testing[ids_smallest_correct, i], bins=50, label="correct (small mass)", histtype="step", normed=True)
#     plt.xlabel("Feature " + str(i))
#     plt.legend(loc="best")
#     if i > 25:
#         plt.xscale("log")
#     else:
#         plt.xscale("linear")


# for i in range(14):
#     ids_z0 = (true_z0 >= bins_plotting[i]) & (true_z0 <= bins_plotting[i + 1])
#     ids_ic = (true_ic >= bins_plotting[i]) & (true_ic <= bins_plotting[i + 1])
#     plt.figure()
#     plt.hist(pred_ic[ids_ic], bins=50, label="z=99", histtype="step", normed=True)
#     plt.hist(pred_z0[ids_z0], bins=50, label="z=0", histtype="step", normed=True)
#
#     plt.axvline(x=bins_plotting[i], color="k")
#     plt.axvline(x=bins_plotting[i + 1], color="k")
#     plt.xlabel("Predicted masses")
#     plt.legend(loc="best")
#     plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/z0_den/loss_lab_nest1000_lr001"
#                 "/vs_ics/z0_vs_ics_bin_" + str(i) + ".pdf")
#
# for i in range(14):
#     ids_RF = (true_z0 >= bins_plotting[i]) & (true_z0 <= bins_plotting[i + 1])
#     # ids_ic = (true_ic >= bins_plotting[i]) & (true_ic <= bins_plotting[i + 1])
#     plt.figure()
#     plt.hist(pred_RF_z0[ids_RF], bins=50, label="RF", histtype="step", normed=True)
#     plt.hist(pred_z0[ids_RF], bins=50, label="GBT", histtype="step", normed=True)
#
#     plt.axvline(x=bins_plotting[i], color="k")
#     plt.axvline(x=bins_plotting[i + 1], color="k")
#     plt.xlabel("Predicted masses")
#     plt.legend(loc="best")
# #     plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/z0_den/loss_lab_nest1000_lr001"
# #                 "/vs_ics/z0_vs_ics_bin_" + str(i) + ".pdf")

