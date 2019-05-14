# predictions AdaBoost for training set for 0.4 features only


# param_grid = {"n_estimators": [500, 100],
#               "learning_rate": [0.001, 0.01, 0.1],
#               "loss": ["linear", "exponential"]
#               }

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from mlhalos import machinelearning as ml
from regression.plots import plotting_functions as pf
import matplotlib.pyplot as plt


def train_gbm(training_features, param_grid={}, cv=True, save=False, path=""):
    X = training_features[:, :-1]
    y = training_features[:, -1]

    if cv is True:
        gbm_base = GradientBoostingRegressor()
        gbm = GridSearchCV(estimator=gbm_base, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    else:
        gbm = GradientBoostingRegressor(**param_grid)

    gbm.fit(X, y)

    if save is True:
        joblib.dump(gbm, path)
    return gbm


def test_gbm(classifier, X_test):
    return classifier.predict(X_test)


def train_and_test_gradboost(training_features, testing_features, param_grid={}, cv=True):
    clf = train_gbm(training_features, param_grid=param_grid, cv=cv)
    y_test = test_gbm(clf, testing_features)
    return clf, y_test


if __name__ == "__main__":
    saving_path = "/share/data2/lls/regression/gradboost/04_only/cv_1/"

    # First load the training set and testing set

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    training_ids = np.load(
        "/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
    log_mass_training = np.log10(halo_mass[training_ids])
    training_features_04 = np.load("/share/data2/lls/regression/adaboost/truth_04_only_2/training_feat_04.npy")

    testing_ids = np.load("/share/data2/lls/regression/adaboost/ic_traj_training/test_ids.npy")
    log_halo_mass_testing = np.log10(halo_mass[testing_ids])
    testing_signal_04_corr = np.load("/share/data2/lls/regression/adaboost/truth_04_only_2/testing_feat_04.npy")

    # train

    cv_i = True

    # param_grid = {"loss": ["huber"],
    #               "learning_rate": [0.2],
    #               "n_estimators": [500, 800],
    #               "max_depth":[5, 100],
    #               "max_features":[0.3]
    #               }

    param_grid = {"loss": ["huber"],
                  "learning_rate": [0.1, 0.15, 0.2],
                  "n_estimators": [800, 1000],
                  "max_depth":[5],
                  "max_features":[0.3, "sqrt"]
                  }

    ada_CV, pred_test = train_and_test_gradboost(training_features_04, testing_signal_04_corr, param_grid=param_grid,
                                                 cv=cv_i)
    np.save(saving_path + "predicted_test_set.npy", pred_test)
    ml.write_to_file_cv_results(saving_path + "cv_results.txt", ada_CV)
    joblib.dump(ada_CV, saving_path + "clf.pkl")

    # predictions

    if cv_i is True:
        alg = ada_CV.best_estimator_
    else:
        alg = ada_CV

    ada_r2_train = np.zeros(len(alg.estimators_), )
    for i, y_pred in enumerate(alg.staged_predict(training_features_04[:, :-1])):
        ada_r2_train[i] = r2_score(log_mass_training, y_pred)

    np.save(saving_path + "r2_train_staged_scores.npy", ada_r2_train)

    ada_r2_test = np.zeros(len(alg.estimators_), )
    for i, y_pred in enumerate(alg.staged_predict(testing_signal_04_corr)):
        ada_r2_test[i] = r2_score(log_halo_mass_testing, y_pred)

    np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)


    ########## PLOT PREDICTIONS ###########

    # pred = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/04_only/predicted_test_set.npy")
    # halo_mass_test = np.load(
    #     "/Users/lls/Documents/mlhalos_files/regression/adaboost/ic_traj_training/halo_mass_test_ids.npy")
    #
    # bins_plotting = np.linspace(halo_mass_test.min(), halo_mass_test.max(), 15, endpoint=True)
    # pf.get_violin_plot(bins_plotting, pred, halo_mass_test, label_distr="0.4 corr feat.")
    # plt.legend(loc="best")
    #
    # pred_RF = np.load(
    #     "/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/predicted_test_set_04_only.npy")
    #
    # true_RF = np.load(
    #     "/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/log_halo_mass_testing.npy")
    #
    # for i in [0, 1, 6, 12, 13]:
    #     ids = np.where((halo_mass_test >= bins_plotting[i]) & (halo_mass_test <= bins_plotting[i + 1]))[0]
    #     ids_RF = np.where((true_RF >= bins_plotting[i]) & (true_RF <= bins_plotting[i + 1]))[0]
    #
    #     plt.figure()
    #     plt.hist(pred_RF[ids_RF], bins=50, color="b", label="RF", histtype="step", normed=True)
    #     plt.hist(pred[ids], bins=50, color="g", label="GBM", histtype="step", normed=True)
    #
    #     plt.axvline(x=bins_plotting[i], color="k", lw=2)
    #     plt.axvline(x=bins_plotting[i + 1], color="k", lw=2)
    #     # plt.title("Bin " + str(i))
    #     plt.legend(loc="best")
    #     plt.subplots_adjust(bottom=0.15)
    #     plt.xlabel("Predicted log mass")
    #     plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/04_only/gbm_vs_RF_bin" +
    #                 str(i) + ".png")

