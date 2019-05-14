"""
In this script, we want to understand how feature importances responde to noise
in the context of regression, multiclass multi-output and multiclass one vs rest

"""


import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from random import shuffle
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def compute_gaussian_noise_features(shape, std):
    n_gaussian = np.zeros(shape)
    for i in range(n_gaussian.shape[1]):
        n_gaussian[:, i] = np.random.normal(loc=0.0, scale=std[i], size=n_gaussian.shape[0])

    return n_gaussian


def get_duplicate_w_fraction_shuffled_samples(array, noise_fraction):
    indices_to_shuffle = np.random.choice(len(array), int(noise_fraction * len(array)), replace=False)
    array_copy = np.copy(array)

    array_to_shuffle = array_copy[indices_to_shuffle]
    shuffle(array_to_shuffle)
    array_copy[indices_to_shuffle] = array_to_shuffle

    print("Fraction of shuffled samples in the duplicate is " + str(np.sum(array_copy != array)/len(array)))

    return array_copy


def get_importances(features, n_estimators=500, max_features=17, min_samples_leaf=1, min_samples_split=2):
    RF = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                               min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                               bootstrap=True, n_jobs=24)
    RF.fit(features[:, :-1], features[:, -1])
    return RF.feature_importances_


def train_and_test_algorithm(training_set, testing_set, n_estimators=500, max_features=17,
                             min_samples_leaf=1, min_samples_split=2):
    RF = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                               min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                               bootstrap=True, n_jobs=24)

    RF.fit(training_set[:, :-1], training_set[:, -1])
    predictions = RF.predict(testing_set)
    return predictions, RF


def scatter_plot_true_vs_predicted(predicted, true, label="ground truth feature"):
    plt.scatter(true, predicted, alpha=0.001, s=0.01, label=label)
    plt.xlabel("True halo mass")
    plt.ylabel("Predicted halo mass")



#
#
# # add DUPLICATES of ground truth
#
# m_repeat = np.tile(log_mass_training, (100, 1)).transpose()
# training_features = np.column_stack((m_repeat, log_mass_training))
# imp_duplicates = get_importances(training_features, n_estimators=500, max_features=17, min_samples_leaf=1,
#                                  min_samples_split=2)
#
#
# # add duplicates of ground truth which contain a fraction of samples whose values are shuffled
# # in this way we have a duplicate which contains some degree of noise
#
# t = np.copy(log_mass_training)
# n = np.linspace(1, 0, num=100, endpoint=False)[::-1]
# for i in range(1, 100):
#     d = get_duplicate_w_fraction_shuffled_samples(log_mass_training, n[i])
#     t = np.column_stack((t, d))
#
# training_features = np.column_stack((t, log_mass_training))
# imp_fraction_shuffled = get_importances(training_features, n_estimators=500, max_features=17, min_samples_leaf=1,
#                                         min_samples_split=2)


# # REGRESSION
#
# training_features = np.column_stack((log_mass_training, n_training, log_mass_training))
# print(training_features.shape)
#
# # cv = True
# # third_features = int((training_features.shape[1] -1)/3)
# # param_grid = {"n_estimators": [1000, 1300],
# #               "max_features": [third_features, "sqrt", 25, 40],
# #               "min_samples_leaf": [5, 15],
# #               #"criterion": ["mse", "mae"],
# #               }
# cv = True
# third_features = int((training_features.shape[1] -1)/3)
# param_grid = {"n_estimators": [500],
#               "max_features": [third_features, "sqrt"],
#               "min_samples_leaf": [5],
#               #"criterion": ["mse", "mae"],
#               }
# param_grid = {"n_estimators": 100,
#               "max_features": "sqrt",
#               "min_samples_leaf": 5,
#               #"criterion": ["mse", "mae"],
#               }
#
# clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
#                      # save=True,
#                      # path=saving_path + "classifier/classifier.pkl",
#                      param_grid=param_grid)
# if cv is True:
#     print(clf.best_estimator)
#     print(clf.algorithm.best_params_)
#     print(clf.algorithm.best_score_)
#
# np.save(saving_path + "f_imp.npy", clf.feature_importances)
#
#
#
#
# # testr
# testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
# halo_mass_testing = np.log10(halo_mass[testing_ids])
#
# X_test = np.column_stack((ics_den_testing, halo_mass_testing))
# y_predicted = clf.algorithm.predict(X_test)
# np.save(saving_path + "predicted_log_halo_mass.npy", y_predicted)