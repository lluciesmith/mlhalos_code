import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from mlhalos import machinelearning as ml
from regression.adaboost import gbm_04_only as gbm_fun
import matplotlib.pyplot as plt
# from regression.feature_importances import noisy_vs_correlated_features as nc
from regression.plots import plotting_functions as pf


if __name__ == "__main__":
    saving_path = "/share/data2/lls/regression/gradboost/07_only/"

    # First load the training set and testing set

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    training_ids = np.load(
        "/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
    log_mass_training = np.log10(halo_mass[training_ids])
    signal_07_corr = np.load("/share/data2/lls/regression/gradboost/07_04_feat/training_feat_07.npy")

    tr_features_07_04 = np.column_stack((signal_07_corr, log_mass_training))

    testing_ids = np.load("/share/data2/lls/regression/adaboost/ic_traj_training/test_ids.npy")
    log_halo_mass_testing = np.log10(halo_mass[testing_ids])
    testing_signal_07_corr = np.load("/share/data2/lls/regression/gradboost/07_04_feat/testing_feat_07.npy")


    # train

    cv_i = True

    # param_grid = {"loss": ["huber"],
    #               "learning_rate": [0.2],
    #               "n_estimators": [500, 800],
    #               "max_depth":[5, 100],
    #               "max_features":[0.3, "sqrt"]
    #               }

    param_grid = {"loss": ["huber"],
                  "learning_rate": [0.1, 0.15, 0.2],
                  "n_estimators": [800, 1000],
                  "max_depth":[5],
                  "max_features":[0.3, "sqrt"]
                  }

    ada_CV, pred_test = gbm_fun.train_and_test_gradboost(tr_features_07_04, testing_signal_07_corr,
                                                         param_grid=param_grid, cv=cv_i)
    np.save(saving_path + "predicted_test_set.npy", pred_test)
    ml.write_to_file_cv_results(saving_path + "cv_results.txt", ada_CV)
    joblib.dump(ada_CV, saving_path + "clf.pkl")

    # predictions

    if cv_i is True:
        alg = ada_CV.best_estimator_
    else:
        alg = ada_CV

    ada_r2_train = np.zeros(len(alg.estimators_), )
    for i, y_pred in enumerate(alg.staged_predict(tr_features_07_04[:, :-1])):
        ada_r2_train[i] = r2_score(log_mass_training, y_pred)

    np.save(saving_path + "r2_train_staged_scores.npy", ada_r2_train)

    ada_r2_test = np.zeros(len(alg.estimators_), )
    for i, y_pred in enumerate(alg.staged_predict(testing_signal_07_corr)):
        ada_r2_test[i] = r2_score(log_halo_mass_testing, y_pred)

    np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)
