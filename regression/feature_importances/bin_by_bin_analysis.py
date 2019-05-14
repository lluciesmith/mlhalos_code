"""
Compare RF prediction with an analytic one for example using linear regression

"""
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from regression.feature_importances import functions_imp_tests as it
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from regression.plots import plotting_functions as pf
from sklearn.linear_model import LinearRegression

# First load the training set and testing set to use for the feature importances tests

training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/training_ids.npy")
log_mass_training = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                            "files/log_halo_mass_training.npy")

testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/testing_ids.npy")
log_halo_mass_testing = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                                "files/log_halo_mass_testing.npy")


############### Compare 0.4 correlated and 0.7 correlated ###############


# training features
dup = np.copy(log_mass_training)
dup1 = np.tile(dup, (50, 1)).transpose()

noise_07 = np.random.normal(0, 1.2, [len(log_mass_training), 50])
signal_07_corr = dup1 + noise_07

noise_04 = np.zeros((len(log_mass_training), 50))
for i in range(50):
    noise_04[:,i] = np.random.normal(0, 2.7, size=len(log_mass_training))

signal_04_corr = dup1 + noise_04

training_features_04_only = np.column_stack((signal_04_corr, log_mass_training))
training_features_all = np.column_stack((signal_07_corr, signal_04_corr, log_mass_training))

# testing features
testing_dup = np.copy(log_halo_mass_testing)
testing_dup1 = np.tile(testing_dup, (50, 1)).transpose()

testing_noise_07 = np.random.normal(0, 1.2, [len(log_halo_mass_testing), 50])
testing_signal_07_corr = testing_dup1 + testing_noise_07

testing_noise_04 = np.random.normal(0, 2.7, [len(log_halo_mass_testing), 50])
testing_signal_04_corr = testing_dup1 + testing_noise_04

testing_features_07_04 = np.column_stack((testing_signal_07_corr, testing_signal_04_corr))


############### RF predictions ###############

pred_04_only, RF_04_only = it.train_and_test_algorithm(training_features_04_only, testing_signal_04_corr,
                                                       n_estimators=100, max_features=10, min_samples_leaf=1,
                                                       min_samples_split=2)

pred_07_04_features, RF_07_04_features = it.train_and_test_algorithm(training_features_all, testing_features_07_04,
                                                                     n_estimators=500, max_features=10,
                                                                     min_samples_leaf=1, min_samples_split=2)
############### LINEAR predictions ###############

reg = LinearRegression().fit(training_features_all[:, :-1], training_features_all[:, -1])
m_reg_07_04 = reg.predict(testing_features_07_04)


reg_04_only = LinearRegression().fit(training_features_04_only[:, :-1], training_features_04_only[:, -1])
m_reg_04_only = reg_04_only.predict(testing_signal_04_corr)


bins_plotting = np.linspace(log_halo_mass_testing.min(), log_halo_mass_testing.max(), 15, endpoint=True)
pf.get_violin_plot(bins_plotting, m_reg_07_04, log_halo_mass_testing)
pf.get_violin_plot(bins_plotting, pred_07_04_features, log_halo_mass_testing)

for i in range(len(bins_plotting) - 1):
    ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing<= bins_plotting[i+1]))[0]

    plt.figure()
    n, b, p = plt.hist(m_reg_04_only[ids], bins=50, color="g", label="Lin. Reg. ($0.4$ feat.)", lw=2,
                       histtype="step", ls="--")
    n1, b1, p1 = plt.hist(pred_04_only[ids], bins=b, color="b", histtype="step", label="RF($0.4$ feat.)", ls="--", lw=2)

    n2, b2, p2 = plt.hist(m_reg_07_04[ids], bins=b, color="g", label="Lin. Reg. ($0.7 + 0.4$ feat.)", histtype="step")
    n3, b3, p3 = plt.hist(pred_07_04_features[ids], bins=b, color="b", histtype="step", label="RF ($0.7 + 0.4$ feat.)")

    plt.axvline(x=bins_plotting[i], color="k", lw=2)
    plt.axvline(x=bins_plotting[i + 1], color="k", lw=2)
    # plt.title("Bin " + str(i))
    plt.legend(loc="best")
    plt.subplots_adjust(top=0.9)
    plt.xlabel("Predicted log mass")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/bin_" + str(i) +
                "_RF_vs_linreg.png")
    plt.clf()

for i in range(len(bins_plotting) - 1):
    ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing <= bins_plotting[i + 1]))[0]
    mid_bin = (bins_plotting[i] + bins_plotting[i + 1])/2
    plt.plot([bins_plotting[i], bins_plotting[i + 1]], [mid_bin, mid_bin], color="k")
    if i==0:
        plt.errorbar(mid_bin-0.05, np.mean(m_reg_07_04[ids]), yerr=np.std(m_reg_07_04[ids]), fmt="o",
                     color="g", label="Lin. Reg.")
        plt.errorbar(mid_bin+0.05, np.mean(pred_07_04_features[ids]), yerr=np.std(pred_07_04_features[ids]), fmt="o",
                     color="b", label="RF")
        plt.errorbar(mid_bin, np.mean(log_halo_mass_testing[ids]), yerr=np.std(log_halo_mass_testing[ids]),
                     fmt="o",
                     color="k", label="true")
    else:
        plt.errorbar(mid_bin-0.05, np.mean(m_reg_07_04[ids]), yerr=np.std(m_reg_07_04[ids]),  fmt="o", color="g")
        plt.errorbar(mid_bin+0.05, np.mean(pred_07_04_features[ids]), yerr=np.std(pred_07_04_features[ids]), fmt="o",
                     color="b")
        plt.errorbar(mid_bin, np.mean(log_halo_mass_testing[ids]), yerr=np.std(log_halo_mass_testing[ids]),
                     fmt="o",
                     color="k")
    plt.legend(loc="best")
    #plt.plot(bins_plotting, bins_plotting, color="k")

for i in range(len(bins_plotting) - 1):
    ids = np.where((log_halo_mass_testing >= bins_plotting[i]) & (log_halo_mass_testing <= bins_plotting[i + 1]))[0]
    mid_bin = (bins_plotting[i] + bins_plotting[i + 1])/2
    plt.plot([bins_plotting[i], bins_plotting[i + 1]], [mid_bin, mid_bin], color="k")
    if i==0:
        plt.errorbar(mid_bin-0.05, np.median(m_reg_04_only[ids]), yerr=np.std(m_reg_04_only[ids]), fmt="o",
                     color="g", label="Lin. Reg.")
        plt.errorbar(mid_bin+0.05, np.mean(pred_04_only[ids]), yerr=np.std(pred_04_only[ids]), fmt="o",
                     color="b", label="RF")
        plt.errorbar(mid_bin, np.mean(log_halo_mass_testing[ids]), yerr=np.std(log_halo_mass_testing[ids]),
                     fmt="o",
                     color="k", label="true")
    else:
        plt.errorbar(mid_bin-0.05, np.median(m_reg_04_only[ids]), yerr=np.std(m_reg_04_only[ids]),  fmt="o", color="g")
        plt.errorbar(mid_bin+0.05, np.mean(pred_04_only[ids]), yerr=np.std(pred_04_only[ids]), fmt="o",
                     color="b")
        plt.errorbar(mid_bin, np.mean(log_halo_mass_testing[ids]), yerr=np.std(log_halo_mass_testing[ids]),
                     fmt="o",
                     color="k")
    plt.legend(loc="best")
    #plt.plot(bins_plotting, bins_plotting, color="k")