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


############### TEST 1 -- add duplicates of the ground truth halo mass ###############

# add DUPLICATES of ground truth

m_repeat = np.tile(log_mass_training, (100, 1)).transpose()
training_features = np.column_stack((m_repeat, log_mass_training))
# imp_duplicates = it.get_importances(training_features, n_estimators=100, max_features=3, min_samples_leaf=1,
#                                       min_samples_split=2)
imp_10 = np.zeros((9, 100))
for i in range(9):
    imp_10[i] = it.get_importances(training_features, n_estimators=100, max_features=33, min_samples_leaf=1,
                                   min_samples_split=2)

imp_500 = np.zeros((9, 100))
for i in range(9):
    imp_500[i] = it.get_importances(training_features, n_estimators=500, max_features=33, min_samples_leaf=1,
                                   min_samples_split=2)

imp_1000 = np.zeros((9, 100))
for i in range(9):
    imp_1000[i] = it.get_importances(training_features, n_estimators=1000, max_features=33, min_samples_leaf=1,
                                   min_samples_split=2)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 7), sharey=True)
ax1.errorbar(range(100), np.mean(imp_10, axis=0), color="r", fmt="o", yerr=np.std(imp_10, axis=0),
             label="100 trees", markersize=5)
ax2.errorbar(range(100), np.mean(imp_500, axis=0), color="b", fmt="o", yerr=np.std(imp_500, axis=0),
             label="500 trees", markersize=5)
ax3.errorbar(range(100), np.mean(imp_1000, axis=0), color="g", fmt="o", yerr=np.std(imp_1000, axis=0),
             label="1000 trees", markersize=5)
ax1.set_xlabel("Feature")
ax2.set_xlabel("Feature")
ax3.set_xlabel("Feature")
ax1.set_ylabel("Importance")
plt.ylim(-0.02, 0.04)
ax1.set_xlim(-1, 100)
ax1.axhline(y=1/100, color="k", ls="--")
ax2.axhline(y=1/100, color="k", ls="--")
ax3.axhline(y=1/100, color="k", ls="--")
ax1.legend(loc="best")
ax2.legend(loc="best")
ax3.legend(loc="best")
plt.subplots_adjust(bottom=0.15,wspace=0)


############### TEST 2 -- add duplicates of the true halo mass which contain noise (reshuffled values) ###############

# add duplicates of ground truth which contain a fraction of samples whose values are shuffled
# in this way we have a duplicate which contains some degree of noise

t = np.copy(log_mass_training)
n = np.linspace(1, 0, num=100, endpoint=False)[::-1]
for i in range(100):
    d = it.get_duplicate_w_fraction_shuffled_samples(log_mass_training, n[i])
    t = np.column_stack((t, d))

corr = np.corrcoef(t.transpose())

training_features = np.column_stack((t, log_mass_training))

# 100 trees, third max features
imp_fraction_shuffled = it.get_importances(training_features, n_estimators=100, max_features=33, min_samples_leaf=1,
                                        min_samples_split=2)

# 500 trees, third max feat
imp_fraction_shuffled_500 = it.get_importances(training_features, n_estimators=500, max_features=33, min_samples_leaf=1,
                                        min_samples_split=2)

# 500 trees, sqrt max feat
imp_fraction_shuffled_500_sqrt_feat = it.get_importances(training_features, n_estimators=500, max_features=10,
                                               min_samples_leaf=1, min_samples_split=2)

# 1000 trees, sqrt max feat
imp_fraction_shuffled_1000_sqrt_feat = it.get_importances(training_features, n_estimators=1000, max_features=10,
                                               min_samples_leaf=1, min_samples_split=2)

# 500 trees, half max feat
imp_fraction_shuffled_500_half_feat = it.get_importances(training_features, n_estimators=500, max_features=50,
                                                         min_samples_leaf=1, min_samples_split=2)

plt.scatter(np.concatenate(([0], n[:-1])), imp_fraction_shuffled, label="100 trees")
plt.scatter(np.concatenate(([0], n[:-1])), imp_fraction_shuffled_500_sqrt_feat, color="b", label="500 trees, "
                                                                                                 "sqrt max feat")
plt.scatter(np.concatenate(([0], n[:-1])), imp_fraction_shuffled_500, color="r", label="500 trees, third max feat")
plt.scatter(np.concatenate(([0], n[:-1])), imp_fraction_shuffled_500_half_feat, color="g", label="500 trees, "
                                                                                                 "half max feat")
plt.scatter(np.concatenate(([0], n[:-1])), imp_fraction_shuffled_1000_sqrt_feat, color="m", label="1000 trees, "
                                                                                                  "sqrt max feat")
plt.xlabel("Fraction of shuffled samples")
plt.ylabel("Importance")
plt.yscale("log")
plt.ylim(10**-7, 1)
plt.legend(loc="best")

# How does the choice of max features influence the predictions?

testing_features = np.copy(log_halo_mass_testing)
n_test = np.linspace(1, 0, num=100, endpoint=False)[::-1]
for i in range(1, 100):
    d_test = it.get_duplicate_w_fraction_shuffled_samples(log_halo_mass_testing, n_test[i])
    testing_features = np.column_stack((testing_features, d_test))

pred_500_third, RF_500_thrid = it.train_and_test_algorithm(training_features, testing_features,
                                                           n_estimators=500, max_features=33, min_samples_leaf=1,
                                                           min_samples_split=2)
pred_500_sqrt, RF_500_sqrt = it.train_and_test_algorithm(training_features, testing_features,
                                                           n_estimators=500, max_features=10, min_samples_leaf=1,
                                                           min_samples_split=2)

mse_sqrt = np.sum((pred_500_sqrt - log_halo_mass_testing)**2)/len(testing_ids)
mse_third = np.sum((pred_500_third - log_halo_mass_testing)**2)/len(testing_ids)

fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
ax1.scatter(log_halo_mass_testing, pred_500_sqrt, alpha=0.1, label="max feat sqrt", color="b")
ax2.scatter(log_halo_mass_testing, pred_500_third, alpha=0.1, label="max feat third", color="g")
ax1.set_ylabel("Predicted (log) halo mass")
ax1.set_xlabel("True (log) halo mass")
ax2.set_xlabel("True (log) halo mass")

ax1.legend(loc="best")
ax2.legend(loc="best")

plt.subplots_adjust(bottom=0.15,wspace=0)


############### TEST 3 -- add Gaussian noise to duplicates of the true halo mass ###############

# training features
dup = np.copy(log_mass_training)
dup1 = np.tile(dup, (50, 1)).transpose()

noise_07 = np.random.normal(0, 1.2, [len(log_mass_training), 50])
signal_07_corr = dup1 + noise_07

noise_04 = np.random.normal(0, 2.7, [len(log_mass_training), 50])
signal_04_corr = dup1 + noise_04

training_features_07_only = np.column_stack((signal_07_corr, log_mass_training))
training_features_all = np.column_stack((signal_07_corr, signal_04_corr, log_mass_training))
corrcoef_07_only = np.corrcoef(training_features_07_only.transpose())
corrcoef_all = np.corrcoef(training_features_all.transpose())

# testing features
testing_dup = np.copy(log_halo_mass_testing)
testing_dup1 = np.tile(testing_dup, (50, 1)).transpose()

testing_noise_07 = np.random.normal(0, 1.2, [len(log_halo_mass_testing), 50])
testing_signal_07_corr = testing_dup1 + testing_noise_07

testing_noise_04 = np.random.normal(0, 2.7, [len(log_halo_mass_testing), 50])
testing_signal_04_corr = testing_dup1 + testing_noise_04

testing_features_07_04 = np.column_stack((testing_signal_07_corr, testing_signal_04_corr))

# predictions
pred_07_only, RF_07_only = it.train_and_test_algorithm(training_features_07_only, testing_signal_07_corr,
                                                       n_estimators=500, max_features=10, min_samples_leaf=1,
                                                       min_samples_split=2)

pred_07_04_features, RF_07_04_features = it.train_and_test_algorithm(training_features_all, testing_features_07_04,
                                                                     n_estimators=500, max_features=10,
                                                                     min_samples_leaf=1, min_samples_split=2)

# plot importances

fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
ax1.scatter(range(50), RF_07_only.feature_importances_, label="$0.7$ correlated features", color="g")
ax2.scatter(range(50), RF_07_04_features.feature_importances_[:50], label="$0.7$ correlated features",
            color="g")
ax2.scatter(range(50, 100), RF_07_04_features.feature_importances_[50:], label="$0.4$ correlated features",
            color="b")
ax1.set_xlabel("Features")
ax2.set_xlabel("Features")
ax1.set_ylabel("Importance")
ax1.set_title("$0.7$ correlated features only")
ax2.set_title("$0.7$ and $0.4$ correlated features")

ax1.legend(loc="best")
ax2.legend(loc="best")
plt.subplots_adjust(bottom=0.15, top=0.88, wspace=0)
plt.colorbar()

# plot predictions
fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
ax1.scatter(log_halo_mass_testing, pred_07_only, alpha=0.1, label="$0.7$ correlated features", color="b")
ax2.scatter(log_halo_mass_testing, pred_07_04_features, alpha=0.1, label="$0.7$ and $0.4$ correlated features", color="b")
ax1.set_ylabel("Predicted (log) halo mass")
ax1.set_xlabel("True (log) halo mass")
ax2.set_xlabel("True (log) halo mass")

ax1.legend(loc="best")
ax2.legend(loc="best")

plt.subplots_adjust(bottom=0.15,wspace=0)


############### TEST 4 -- add Gaussian noise of various sigma to duplicates of the true halo mass ###############

# training features
dup = np.copy(log_mass_training)
dup1 = np.tile(dup, (100, 1)).transpose()

noise_level = np.linspace(1.2, 2.7, num=100, endpoint=True)
noise = np.random.normal(np.zeros((100,)), noise_level, [len(log_mass_training), 100])

signal_training = dup1 + noise

# testing features

testing_dup = np.copy(log_halo_mass_testing)
testing_dup1 = np.tile(testing_dup, (100, 1)).transpose()

noise_testing = np.random.normal(np.zeros((100,)), noise_level, [len(log_halo_mass_testing), 100])

signal_testing = testing_dup1 + noise_testing

# predictions
training_features = np.column_stack((signal_training, log_mass_training))

pred, RF = it.train_and_test_algorithm(training_features, signal_testing, n_estimators=500, max_features=10,
                                       min_samples_leaf=1, min_samples_split=2)


# plot importances

fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
ax1.scatter(noise_level, RF.feature_importances_, color="b")
ax2.scatter(range(50), RF_07_04_features.feature_importances_[:50], label="$0.7$ correlated features",
            color="g")
ax2.scatter(range(50, 100), RF_07_04_features.feature_importances_[50:], label="$0.4$ correlated features",
            color="b")
ax1.set_xlabel("Features")
ax2.set_xlabel("Features")
ax1.set_ylabel("Importance")
ax1.set_title("$0.7$ correlated features only")
ax2.set_title("$0.7$ and $0.4$ correlated features")

ax1.legend(loc="best")
ax2.legend(loc="best")
plt.subplots_adjust(bottom=0.15, top=0.88, wspace=0)
plt.colorbar()

# plot predictions
fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
ax1.scatter(log_halo_mass_testing, pred_07_only, alpha=0.1, label="$0.7$ correlated features", color="b")
ax2.scatter(log_halo_mass_testing, pred_07_04_features, alpha=0.1, label="$0.7$ and $0.4$ correlated features", color="b")
ax1.set_ylabel("Predicted (log) halo mass")
ax1.set_xlabel("True (log) halo mass")
ax2.set_xlabel("True (log) halo mass")

ax1.legend(loc="best")
ax2.legend(loc="best")

plt.subplots_adjust(bottom=0.15,wspace=0)


