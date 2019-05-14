# predictions AdaBoost for training set for 0.4 features only


# param_grid = {"n_estimators": [500, 100],
#               "learning_rate": [0.001, 0.01, 0.1],
#               "loss": ["linear", "exponential"]
#               }

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from mlhalos import machinelearning as ml


def train_adaboost(training_features, param_grid={}, cv=True, save=False, path=""):
    X = training_features[:, :-1]
    y = training_features[:, -1]

    base_estimator = DecisionTreeRegressor(max_depth=5)
    if cv is True:
        ada_b = AdaBoostRegressor(base_estimator=base_estimator)
        ada = GridSearchCV(estimator=ada_b, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    else:
        ada = AdaBoostRegressor(base_estimator=base_estimator, **param_grid)
    ada.fit(X, y)

    if save is True:
        joblib.dump(ada, path)
    return ada


def test_adaboost(classifier, X_test):
    return classifier.predict(X_test)


def train_and_test_adaboost(training_features, testing_features, param_grid={}, cv=True):
    clf = train_adaboost(training_features, param_grid=param_grid, cv=cv)
    y_test = test_adaboost(clf, testing_features)
    return clf, y_test


if __name__ == "__main__":
    saving_path = "/share/data2/lls/regression/adaboost/truth_04_only_2/"

    # First load the training set and testing set

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    training_ids = np.load(
        "/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
    log_mass_training = np.log10(halo_mass[training_ids])
    training_features_04 = np.load(saving_path + "training_feat_04.npy")

    testing_ids = np.load("/share/data2/lls/regression/adaboost/ic_traj_training/test_ids.npy")
    log_halo_mass_testing = np.log10(halo_mass[testing_ids])
    testing_signal_04_corr = np.load(saving_path + "testing_feat_04.npy")

    # train

    cv_i = False

    # param_grid = {"n_estimators": [300, 500, 800],
    #               "learning_rate": [0.01, 0.05, 0.1, 0.2],
    #               "base_estimator__max_depth": [5, 8]
    #               }
    param_grid = {"n_estimators": 1200,
                  "learning_rate": 0.1}

    ada_CV, pred_test = train_and_test_adaboost(training_features_04, testing_signal_04_corr,
                                                param_grid=param_grid, cv=cv_i)
    np.save(saving_path + "predicted_test_set_n1200_lr01.npy", pred_test)
    # ml.write_to_file_cv_results(saving_path + "cv_results.txt", ada_CV)
    joblib.dump(ada_CV, saving_path + "clf_n1200_lr01.pkl")

    # predictions

    if cv_i is True:
        alg = ada_CV.best_estimator_
    else:
        alg = ada_CV

    ada_r2_train = np.zeros(len(alg.estimators_), )
    for i, y_pred in enumerate(alg.staged_predict(training_features_04[:,:-1])):
        ada_r2_train[i] = r2_score(log_mass_training, y_pred)

    np.save(saving_path + "r2_train_staged_scoresn1200_lr01.npy", ada_r2_train)

    ada_r2_test = np.zeros(len(alg.estimators_), )
    for i, y_pred in enumerate(alg.staged_predict(testing_signal_04_corr)):
        ada_r2_test[i] = r2_score(log_halo_mass_testing, y_pred)

    np.save(saving_path + "r2_test_staged_scoresn1200_lr01.npy", ada_r2_test)


