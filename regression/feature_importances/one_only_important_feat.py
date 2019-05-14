"""


"""
import sys
# sys.path.append("/Users/lls/Documents/mlhalos_code")
sys.path.append("/home/lls//mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from regression.adaboost import gbm_04_only as gbm_fun
from mlhalos import machinelearning as ml
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# First load the training set and testing set to use for the feature importances tests

# training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/training_ids.npy")
# log_mass_training = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
#                             "files/log_halo_mass_training.npy")
#
# testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/testing_ids.npy")
# log_halo_mass_testing = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
#                                 "files/log_halo_mass_testing.npy")


halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
log_mass_training = np.log10(halo_mass[training_ids])

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
log_halo_mass_testing = np.log10(halo_mass[testing_ids])

############### Compare 0.4 correlated and 0.7 correlated ###############

saving_path = "/share/data2/lls/regression/gradboost/rel_feature_high_masses/"

# rel_training = (log_mass_training >= 11.5) & (log_mass_training <= 12.5)
# rel_testing = (log_halo_mass_testing >= 11.5) & (log_halo_mass_testing <= 12.5)
# rel_training = (log_mass_training <= 11.5)
# rel_testing = (log_halo_mass_testing <= 11.5)
rel_training = (log_mass_training >= 12.5)
rel_testing = (log_halo_mass_testing >= 12.5)

# training features

dup = np.copy(log_mass_training)
training_06_corr = np.zeros(len(log_mass_training), )
training_06_corr[rel_training] = dup[rel_training] + np.random.normal(0, 0.3, len(log_mass_training[rel_training]))
training_06_corr[~rel_training] = dup[~rel_training] + np.random.normal(0, 11, len(log_mass_training[~rel_training]))

dup = np.copy(log_mass_training)
dup1 = np.tile(dup, (49, 1)).transpose()
noise_02 = np.random.normal(0, 11, [len(log_mass_training), 49])
training_02_corr = dup1 + noise_02

training_features = np.column_stack((training_02_corr[:, :25], training_06_corr, training_02_corr[:, 25:],
                                     log_mass_training))

np.save(saving_path + "training_feat.npy", training_features)

# testing features

dup_test = np.copy(log_halo_mass_testing)
testing_06_corr = np.zeros(len(log_halo_mass_testing), )
testing_06_corr[rel_testing] = dup_test[rel_testing] + \
                                np.random.normal(0, 0.3, len(log_halo_mass_testing[rel_testing]))
testing_06_corr[~rel_testing] = dup_test[~rel_testing] + \
                                 np.random.normal(0, 11, len(log_halo_mass_testing[~rel_testing]))

dup_t = np.copy(log_halo_mass_testing)
dup1 = np.tile(dup_t, (49, 1)).transpose()
test_noise_02 = np.random.normal(0, 11, [len(log_halo_mass_testing), 49])
testing_02_features = dup1 + test_noise_02

testing_features = np.column_stack((testing_02_features[:, :25], testing_06_corr, testing_02_features[:, 25:]))

np.save(saving_path + "testing_feat.npy", testing_features)

# predictions
cv_i = True

param_grid = {"loss": ["huber"],
              "learning_rate": [0.1],
              "n_estimators": [800],
              "max_depth": [5],
              "max_features": [0.3, 0.2]
              }
param_grid = {"loss": "huber",
              "learning_rate": 1,
              "n_estimators": 800,
              "max_depth": 5,
              "max_features": 0.2
              }

gbm, pred_test = gbm_fun.train_and_test_gradboost(training_features, testing_features, param_grid=param_grid,
                                                  cv=False)

np.save(saving_path + "predicted_test_set.npy", pred_test)
joblib.dump(gbm, saving_path + "clf.pkl")
np.save(saving_path + "importances.pdf", gbm.best_estimator_.feature_importances_)

ml.write_to_file_cv_results(saving_path + "cv_results.txt", gbm)

# predictions

if cv_i is True:
    alg = gbm.best_estimator_
else:
    alg = gbm

ada_r2_train = np.zeros(len(alg.estimators_), )
for i, y_pred in enumerate(alg.staged_predict(training_features[:, :-1])):
    ada_r2_train[i] = r2_score(log_mass_training, y_pred)

np.save(saving_path + "r2_train_staged_scores.npy", ada_r2_train)

ada_r2_test = np.zeros(len(alg.estimators_), )
for i, y_pred in enumerate(alg.staged_predict(testing_features)):
    ada_r2_test[i] = r2_score(log_halo_mass_testing, y_pred)

np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)


# Given predictions and true masses, print the bias and variance in the predictions and the variance in the feaures
# for the mass bins.

true = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/true_halo_mass.npy")
log_true = np.log10(true)

bin_small = (log_true <= 11.5)
bin_mid = (log_true >= 11.5) & (log_true <= 12.5)
bins_high = (log_true >= 12.5)

# Case of info in small bin
test_feat_small = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/rel_feature_small_masses"
                          "/testing_feat.npy")
pred_rel_small =  np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/rel_feature_small_masses/"
                          "predicted_test_set.npy")

def plot_predictions_vs_features_three_mass_bins(features, predictions, title=None):
    bin_small = (log_true <= 11.5)
    bin_mid = (log_true >= 11.5) & (log_true <= 12.5)
    bins_high = (log_true >= 12.5)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(10,6), sharey=True, sharex=True)
    ax1.hist(features[bin_small, 25], bins=50, histtype="step", density=True, label="feature")
    ax1.hist(predictions[bin_small],  bins=50, histtype="step", label="predicted masses", density=True)
    #plt.legend(loc="best")
    fig.subplots_adjust(wspace=0)
    ax1.axvline(x=log_true.min(), color="k")
    ax1.axvline(x=11.5, color="k")

    ax2.hist(features[bin_mid, 25], bins=50, histtype="step", density=True, label="feature")
    ax2.hist(predictions[bin_mid],bins=50, histtype="step", label="predicted masses", density=True)
    ax2.set_xlim(9, 15)
    ax2.axvline(x=12.5, color="k")
    ax2.axvline(x=11.5, color="k")

    ax3.hist(features[bins_high, 25], bins=50, histtype="step", density=True, label="feature")
    ax3.hist(predictions[bins_high],  bins=50, histtype="step", label="predicted masses", density=True)
    ax3.set_xlim(9, 15)
    ax3.axvline(x=log_true.max(), color="k")
    ax3.axvline(x=12.5, color="k")

    plt.legend(loc=(-1.5, 0.8))
    if title is not None:
        ax2.set_title(title)
        fig.subplots_adjust(top=0.9)

    ax2.set_xlabel("Log mass")


