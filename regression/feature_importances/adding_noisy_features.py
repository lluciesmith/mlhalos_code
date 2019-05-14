"""


"""
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from regression.feature_importances import functions_imp_tests as it
from random import shuffle
import matplotlib.pyplot as plt


# First load the training set and testing set to use for the feature importances tests

training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/training_ids.npy")
log_mass_training = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                            "files/log_halo_mass_training.npy")

testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/testing_ids.npy")
log_halo_mass_testing = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                                "files/log_halo_mass_testing.npy")


############### TEST 1 -- train on ground truth halo mass ###############

training_set = np.column_stack((log_mass_training, log_mass_training))
predictions_test_set, RF = it.train_and_test_algorithm(training_set,
                                                       log_halo_mass_testing.reshape((len(testing_ids), 1)),
                                                       n_estimators=100, max_features=1)

it.scatter_plot_true_vs_predicted(predictions_test_set, log_halo_mass_testing, label="ground truth feature")


############### TEST 2 -- add noise to the ground truth halo mass ###############

# add Gaussian noise

number_of_features = 50

shape_features = (len(training_ids), number_of_features)
# gaussian_noise_features = it.compute_gaussian_noise_features(shape_features, std=np.arange(1, number_of_features + 1))
gaussian_noise_features = np.random.normal(np.zeros((50,)), np.linspace(0, 1, 50), [len(training_ids), 50])

training_features = np.column_stack((log_mass_training, gaussian_noise_features, log_mass_training))
imp_gaussian_noise = it.get_importances(training_features, n_estimators=500, max_features=7, min_samples_leaf=1,
                                        min_samples_split=2)
del training_features

plt.scatter(range(51)[1:], imp_gaussian_noise[1:], color="b", label="Gaussian noise")
plt.scatter(0, imp_gaussian_noise[0], color="r", label="True log mass")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.legend(loc="best")
plt.yscale("log")




# add noise from the true log mass distribution

num_noisy_features = 50
n_true_distr = np.zeros((len(log_mass_training), num_noisy_features))

for i in range(num_noisy_features):
    shuffled_halo_mass = np.copy(log_mass_training)
    shuffle(shuffled_halo_mass)
    n_true_distr[:, i] = shuffled_halo_mass

training_features = np.column_stack((log_mass_training, n_true_distr, log_mass_training))
imp_noise_true_distr = it.get_importances(training_features, n_estimators=500, max_features=17, min_samples_leaf=1,
                                       min_samples_split=2)

plt.scatter(range(51)[1:], imp_noise_true_distr[1:], color="b", label="Noise from true mass distribution")
plt.scatter(0, imp_noise_true_distr[0], color="r", label="True log mass")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.legend(loc="best")
plt.yscale("log")
