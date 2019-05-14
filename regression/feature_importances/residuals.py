"""
Fit a RF model and then plot for each mass bin, the predicted vs actual values in the training set.

"""

import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from regression.feature_importances import functions_imp_tests as it
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from regression.plots import plotting_functions as pf


# First load the training set and testing set to use for the feature importances tests

training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/training_ids.npy")
log_mass_training = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                            "files/log_halo_mass_training.npy")

testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/testing_ids.npy")
log_halo_mass_testing = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                                "files/log_halo_mass_testing.npy")


# training features
train_feat_04_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                             "/files/training_feat_04_correlation.npy")
train_feat_07_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                             "/files/training_feat_07_correlation.npy")

test_feat_04_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                            "/files/testing_feat_04_correlation.npy")
test_feat_07_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                            "/files/testing_feat_07_correlation.npy")

# training_features_all = np.column_stack((train_feat_07_corr, train_feat_04_corr, log_mass_training))
# testing_features_all = np.column_stack((test_feat_07_corr, test_feat_04_corr))
training_features_04_only = np.column_stack((train_feat_04_corr, log_mass_training))
testing_features_04_only = test_feat_04_corr

# predictions
pred_test_set, RF_trained = it.train_and_test_algorithm(training_features_04_only, testing_features_04_only,
                                                                     n_estimators=500, max_features=10,
                                                                     min_samples_leaf=1, min_samples_split=2)

np.save("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/predicted_test_set_04_only.npy",
        pred_test_set)

# Method 1 from Zhang and Lu (2012) -- train RF to predict residuals given X,y training data

pred_training = RF_trained.predict(training_features_04_only[:, :-1])
res = log_mass_training - pred_training

RF_res = RandomForestRegressor(n_estimators=200, max_features=10, min_samples_leaf=1, min_samples_split=2)
RF_res.fit(np.column_stack((training_features_04_only[:, :-1], pred_training)), res)

# check that you get perfect prediction when you plot log_mass_training vs res
# check what you get when you plot log_mass_training vs pred_training + RF_res.predict(training_features_04_only)

bins_plotting = np.linspace(log_mass_training.min(), log_mass_training.max(), 15, endpoint=True)

y_bc_training = pred_training + RF_res.predict(training_features_04_only)
pf.get_violin_plot(bins_plotting, pred_training + RF_res.predict(training_features_04_only), log_mass_training)

res_test_set = RF_res.predict(np.column_stack((testing_features_04_only, pred_test_set)))
y_bc = pred_test_set + res_test_set
pf.get_violin_plot(bins_plotting, y_bc, log_halo_mass_testing)

# Liaw & Wiener (2009) -- fit a linear regression model

reg = LinearRegression()
reg.fit(pred_training.reshape(-1, 1), log_mass_training)
y_linreg_bc = reg.predict(pred_test_set.reshape(-1, 1))


# Compare histograms for corrected and non-corrected predictions

bins_plotting = np.linspace(log_halo_mass_testing.min(), log_halo_mass_testing.max(), 15, endpoint=True)

for i in [0,6, 13]:
# for i in range(len(bins_plotting) - 1):
    ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing<= bins_plotting[i+1]))[0]

    plt.figure()
    n, b, p = plt.hist(y_bc[ids], bins=30, color="g", label="RF", histtype="step")
    n1, b1, p1 = plt.hist(pred_test_set[ids], bins=b, color="b", histtype="step", label="RF + BC RF")
    n1, b1, p1 = plt.hist(y_linreg_bc[ids], bins=b, color="r", histtype="step", label="RF + BC LR")

    plt.axvline(x=bins_plotting[i], color="k", lw=2)
    plt.axvline(x=bins_plotting[i + 1], color="k", lw=2)
    # plt.title("Bin " + str(i))
    plt.legend(loc="best")
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("Predicted log mass")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/bias_correction"
                "/bc_methods_bin_" + str(
        i) + ".png")


# SCATTER PLOTS EACH BIN

# training data
res_training = RF_res.predict(np.column_stack((training_features_all[:,:-1], pred_training)))

for i in range(len(bins_plotting) - 1):
    ids_training = np.where((log_mass_training >= bins_plotting[i]) & (log_mass_training <= bins_plotting[i + 1]))[0]

    plt.figure()
    plt.xlabel("True mass")
    plt.ylabel("Predicted mass")
    plt.title("Training data predictions")
    plt.subplots_adjust(top=0.9)

    plt.scatter(log_mass_training[ids_training], reg.predict(pred_training[ids_training].reshape(-1, 1)), color="b",
                label="RF + lin.reg. bias")
    plt.scatter(log_mass_training[ids_training], pred_training[ids_training] + res_training[ids_training], color="g",
                label="RF + RF bias corr")

    plt.plot(log_mass_training[ids_training], log_mass_training[ids_training], color="k")
    plt.legend(loc=2)
    plt.savefig(
        "/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/bias_correction"
        "/bias_corr_on_training_data_bin_" +
        str(i) + ".png")

for i in range(len(bins_plotting) - 1):
    ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing <= bins_plotting[i + 1]))[0]

    plt.figure()
    plt.xlabel("True mass")
    plt.ylabel("Predicted mass")
    plt.title("Testing data predictions")
    plt.subplots_adjust(top=0.9)

    plt.scatter(log_halo_mass_testing[ids], reg.predict(pred_test_set[ids].reshape(-1, 1)), color="b",
                label="RF + lin.reg. bias")
    plt.scatter(log_halo_mass_testing[ids], pred_test_set[ids] + res_test_set[ids], color="g",
                label="RF + RF bias corr")

    plt.plot(log_halo_mass_testing[ids], log_halo_mass_testing[ids], color="k")
    plt.legend(loc=2)
    plt.savefig(
        "/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
        "/bias_correction/bias_corr_on_testing_data_bin_" +
        str(i) + ".png")

for i in range(len(bins_plotting) - 1):
    ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing <= bins_plotting[i + 1]))[0]
    mid_bin = (bins_plotting[i] + bins_plotting[i + 1])/2
    plt.plot([bins_plotting[i], bins_plotting[i + 1]], [mid_bin, mid_bin], color="k")
    r = reg.predict(pred_test_set[ids].reshape(-1, 1))
    if i==0:
        plt.errorbar(mid_bin-0.05, np.mean(y_linreg_bc[ids]), yerr=np.std(y_linreg_bc[ids]), fmt="o",
                     color="b", label="RF + lin.reg. bias")
        plt.errorbar(mid_bin, np.mean(y_bc[ids]), yerr=np.std(y_bc[ids]), fmt="o",
                     color="k", label="RF + RF bias corr")
        plt.errorbar(mid_bin+0.05, np.mean(pred_test_set[ids]),
                     yerr=np.std(pred_test_set[ids]), fmt="o",
                     color="g", label="true")
    else:
        plt.errorbar(mid_bin-0.05, np.mean(r), yerr=np.std(r), fmt="o",
                     color="b")
        plt.errorbar(mid_bin, np.mean(y_bc[ids]), yerr=np.std(y_bc[ids]), fmt="o",
                     color="k")
        plt.errorbar(mid_bin+0.05, np.mean(pred_test_set[ids]),
                     yerr=np.std(pred_test_set[ids]), fmt="o",
                     color="g")
    plt.legend(loc="best")