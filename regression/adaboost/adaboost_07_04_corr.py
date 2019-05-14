
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from mlhalos import machinelearning as ml

saving_path = "/share/data2/lls/regression/adaboost/truth_07_04_corr_2/"

# First load the training set and testing set

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
log_mass_training = np.log10(halo_mass[training_ids])

testing_ids = np.load("/share/data2/lls/regression/adaboost/ic_traj_training/test_ids.npy")
log_halo_mass_testing = np.log10(halo_mass[testing_ids])


# training features
dup = np.copy(log_mass_training)
dup1 = np.tile(dup, (50, 1)).transpose()

noise_07 = np.random.normal(0, 1.2, [len(log_mass_training), 50])
signal_07_corr = dup1 + noise_07

noise_04 = np.random.normal(0, 2.7, [len(log_mass_training), 50])
signal_04_corr = dup1 + noise_04

training_features_04_only = np.column_stack((signal_04_corr, log_mass_training))
training_features_all = np.column_stack((signal_07_corr, signal_04_corr))

# testing features
testing_dup = np.copy(log_halo_mass_testing)
testing_dup1 = np.tile(testing_dup, (50, 1)).transpose()

testing_noise_07 = np.random.normal(0, 1.2, [len(log_halo_mass_testing), 50])
testing_signal_07_corr = testing_dup1 + testing_noise_07

testing_noise_04 = np.random.normal(0, 2.7, [len(log_halo_mass_testing), 50])
testing_signal_04_corr = testing_dup1 + testing_noise_04

testing_features_07_04 = np.column_stack((testing_signal_07_corr, testing_signal_04_corr))

# predictions AdaBoost for training set of 0.7 features + 0.4 features


param_grid = {"n_estimators": [500, 800, 1000],
              "learning_rate": [0.05, 0.1, 0.15],
              # "loss": ["linear", "exponential"]
              }
base_estimator = DecisionTreeRegressor(max_depth=5)
ada_CV = GridSearchCV(estimator=AdaBoostRegressor(base_estimator=base_estimator), param_grid=param_grid,
                      cv=3, verbose=2, n_jobs=-1, scoring="r2")
ada_CV.fit(training_features_all, log_mass_training)


ml.write_to_file_cv_results(saving_path + "cv_results.txt", ada_CV)
joblib.dump(ada_CV, saving_path + "clf.pkl")


# predictions

pred_test = ada_CV.predict(testing_features_07_04)
np.save(saving_path + "predicted_test_set.npy", pred_test)

ada_r2_train = np.zeros(len(ada_CV.best_estimator_.estimators_),)
for i, y_pred in enumerate(ada_CV.best_estimator_.staged_predict(training_features_all)):
    ada_r2_train[i] = r2_score(log_mass_training, y_pred)

np.save(saving_path + "r2_train_staged_scores.npy", ada_r2_train)

ada_r2_test = np.zeros(len(ada_CV.best_estimator_.estimators_),)
for i, y_pred in enumerate(ada_CV.best_estimator_.staged_predict(testing_features_07_04)):
    ada_r2_test[i] = r2_score(log_halo_mass_testing, y_pred)

np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)


# fig, ax = plt.subplots()
# ax.plot(np.arange(n_estimators), ada_r2_test,
#         label='Test Error',
#         color='red')
# ax.plot(np.arange(n_estimators), ada_r2_train,
#         label='Train Error',
#         color='blue')
#
# ax.set_xlabel('n estimators')
# ax.set_ylabel('r$^2$ score')
# plt.legend(loc="best")
#
#
#
# for i in range(len(bins_plotting) - 1):
#     ids_ada = np.where((log_halo_mass_testing>= bins_plotting[i]) & (log_halo_mass_testing<= bins_plotting[i+1]))[0]
#     plt.figure()
#     n, b, p = plt.hist(pred_test[ids_ada], histtype="step", bins=25, label="ada")
#     plt.hist(log_halo_mass_testing[ids_ada], histtype="step", bins=10, label="true", color="k")
#     plt.axvline(x=bins_plotting[i], color="k", lw=2)
#     plt.axvline(x=bins_plotting[i + 1], color="k", lw=2)
#
# for i in range(len(bins_plotting) - 1):
#     ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing<= bins_plotting[i+1]))[0]
#
#     plt.figure()
#     n, b, p = plt.hist(m_reg_04_only[ids], bins=50, color="g", label="Lin. Reg. ($0.4$ feat.)", lw=2,
#                        histtype="step", ls="--")
#     n1, b1, p1 = plt.hist(pred_04_only[ids], bins=b, color="b", histtype="step", label="RF($0.4$ feat.)", ls="--", lw=2)
#
#     n2, b2, p2 = plt.hist(m_reg_07_04[ids], bins=b, color="g", label="Lin. Reg. ($0.7 + 0.4$ feat.)", histtype="step")
#     n3, b3, p3 = plt.hist(pred_07_04_features[ids], bins=b, color="b", histtype="step", label="RF ($0.7 + 0.4$ feat.)")
#
#     plt.axvline(x=bins_plotting[i], color="k", lw=2)
#     plt.axvline(x=bins_plotting[i + 1], color="k", lw=2)
#     # plt.title("Bin " + str(i))
#     plt.legend(loc="best")
#     plt.subplots_adjust(top=0.9)
#     plt.xlabel("Predicted log mass")


### Compare histograms to RF predictions

pred_RF = np.load("/Users.")
log_halo_mass_testing

for i in range(len(bins_plotting) - 1):
    ids_ada = np.where((ada_true >= bins_plotting[i]) & (ada_true <= bins_plotting[i + 1]))[0]
    ids_RF = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing <= bins_plotting[i + 1]))[0]
    plt.figure()
    n, b, p = plt.hist(pred_04_only[ids_RF], bins=50, color="g", label="RF", lw=2, normed=True,
                       histtype="step", ls="--")
    n1, b1, p1 = plt.hist(ada_04_only[ids_ada], bins=b, color="b", histtype="step", label="Adaboost", ls="--",
                          lw=2, normed=True)

    # n2, b2, p2 = plt.hist(m_reg_07_04[ids], bins=b, color="g", label="Lin. Reg. ($0.7 + 0.4$ feat.)", histtype="step")
    # n3, b3, p3 = plt.hist(pred_07_04_features[ids], bins=b, color="b", histtype="step", label="RF ($0.7 + 0.4$ feat.)")

    plt.axvline(x=bins_plotting[i], color="k", lw=2)
    plt.axvline(x=bins_plotting[i + 1], color="k", lw=2)
    # plt.title("Bin " + str(i))
    plt.legend(loc="best")
    plt.subplots_adjust(top=0.91)
    plt.xlabel("Predicted log mass")


