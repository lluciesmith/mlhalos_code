
"""
Test gradboost on z=99 density features + z=99 shear features
Test gradboost on z=99 shear only features

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from regression.adaboost import gbm_04_only as gbm_fun
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

saving_path_shear_plus_traj = "/share/data2/lls/regression/gradboost/randomly_sampled_training/shear/shear_and_ic_traj/"
# saving_path_shear_only = "/share/data2/lls/regression/gradboost/randomly_sampled_training/shear/shear_only/"
# features_path = "/share/data2/lls/features_w_periodicity_fix/"
#
# # data
#
# training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006"
#                        "/training_ids.npy")
# testing_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006"
#                        "/testing_ids.npy")
#
# halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
# traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")
# den_sub_ell = np.lib.format.open_memmap(features_path + "density_subtracted_ellipticity.npy", mode="r",
#                                         shape=(256**3, 50))
# den_sub_prol = np.lib.format.open_memmap(features_path + "density_subtracted_prolateness.npy", mode="r",
#                                          shape=(256**3, 50))
#
# traj_training = traj[training_ids]
# ell_training = den_sub_ell[training_ids]
# prol_training = den_sub_prol[training_ids]
# log_halo_training = np.log10(halo_mass[training_ids])
#
# training_features = np.column_stack((traj_training, ell_training, prol_training, log_halo_training))
# np.save(saving_path_shear_plus_traj + "training_features_plus_truth.npy", training_features)
#
# traj_testing = traj[testing_ids]
# ell_testing = den_sub_ell[testing_ids]
# prol_testing = den_sub_prol[testing_ids]
# log_halo_testing = np.log10(halo_mass[testing_ids])
# testing_features = np.column_stack((traj_testing, ell_testing, prol_testing))
# np.save(saving_path_shear_plus_traj + "testing_features.npy", testing_features)
# np.save(saving_path_shear_plus_traj + "log_halo_testing_set.npy", log_halo_testing)
#
training_features = np.load(saving_path_shear_plus_traj + "training_features_plus_truth.npy")
testing_features = np.load(saving_path_shear_plus_traj + "testing_features.npy")

# training on ic traj + shear z=99

# param_grid = {"loss": "lad", "learning_rate": 0.06, "n_estimators": 2000,  "max_depth": 5,  "max_features":"sqrt"}
param_grid = {"loss": "lad", "learning_rate": 0.08, "n_estimators": 1500,  "max_depth": 5,  "max_features":"sqrt"}

clf_traj_shear, pred_test = gbm_fun.train_and_test_gradboost(training_features, testing_features, param_grid=param_grid,
                                                             cv=False)

np.save(saving_path_shear_plus_traj + "nest_1500_lr008/predicted_test_set.npy", pred_test)
joblib.dump(clf_traj_shear, saving_path_shear_plus_traj + "nest_1500_lr008/clf.pkl")
np.save(saving_path_shear_plus_traj + "nest_1500_lr008/importances.npy", clf_traj_shear.feature_importances_)

# # training on shear z=99 only
#
# param_grid = {"loss": "lad", "learning_rate": 0.06, "n_estimators": 2000,  "max_depth": 5,  "max_features":"sqrt"}
#
# training_features = np.column_stack((ell_training, prol_training, log_halo_training))
# testing_features = np.column_stack((ell_testing, prol_testing))
# clf_shear_only, pred_test = gbm_fun.train_and_test_gradboost(training_features, testing_features, param_grid=param_grid,
#                                                              cv=False)
#
# np.save(saving_path_shear_only + "predicted_test_set.npy", pred_test)
# joblib.dump(clf_shear_only, saving_path_shear_only + "clf.pkl")
# np.save(saving_path_shear_only + "importances.npy", clf_shear_only.feature_importances_)
#
#
#
# ############### PLOTS #################
#

path_z99 = "/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj/nest_2000_lr006/"
pred_z99 = np.load(path_z99 + "predicted_test_set.npy")
true_z99 = np.load(path_z99 + "log_halo_testing_set.npy")
mae_bins_ic = np.load(path_z99 + "mae_bins.npy")

path_shear = "/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/shear/shear_and_ic_traj/"
pred_shear_traj = np.load(path_shear + "predicted_test_set.npy")
mae_bins_shear = np.load(path_shear + "mae_bins.npy")

mae_all_shear = np.load(path_shear + "mae_all.npy")
mae_all_den = np.load(path_z99 + "mae_all.npy")

mean_mae_ic = np.mean(mae_bins_ic, axis=0)
std_mae_ic = np.std(mae_bins_ic, axis=0)

mean_mae_shear = np.mean(mae_bins_shear, axis=0)
std_mae_shear = np.std(mae_bins_shear, axis=0)

bins_plotting = np.linspace(true_z99.min(), true_z99.max(), 15, endpoint=True)
for i in range(14):
    ids_gbt = (true_z99 >= bins_plotting[i]) & (true_z99 <= bins_plotting[i + 1])

    # mae_ic = mean_absolute_error(true_z99[ids_gbt], pred_z99[ids_gbt])
    # mae_shear = mean_absolute_error(true_z99[ids_gbt], pred_shear_traj[ids_gbt])

    plt.figure()
    plt.hist(pred_z99[ids_gbt], bins=50, label="den \n (mae = %.3f" % mean_mae_ic[i] + " $\pm$ %.3f)" %
                                                                                           std_mae_ic[i],
             histtype="step", density=True)
    plt.hist(pred_shear_traj[ids_gbt], bins=50, label="shear+den \n (mae = %.3f" % mean_mae_shear[i] + " $\pm$ %.3f)" %
                                                                                                        std_mae_shear[i],
             histtype="step", density=True)

    plt.axvline(x=bins_plotting[i], color="k")
    plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlim(10,15)
    plt.xlabel("$\log (M_{\mathrm{predicted}} / \mathrm{M}_{\odot}) $")
    plt.legend(loc="best", fontsize=14)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training"
                "/shear/shear_and_ic_traj/predictions_bin_" + str(i) + ".png")

for i in range(14):
    ids_gbt = (true_z99 >= bins_plotting[i]) & (true_z99 <= bins_plotting[i + 1])

    mae_ic = mean_absolute_error(true_z99[ids_gbt], pred_1500[ids_gbt])
    mae_shear = mean_absolute_error(true_z99[ids_gbt], pred_shear_traj[ids_gbt])

    plt.figure()
    plt.hist(pred_1500[ids_gbt], bins=50, label="den \n (mae = %.3f" % mae_ic,
             histtype="step", density=True)
    plt.hist(pred_shear_traj[ids_gbt], bins=50, label="shear+den \n (mae = %.3f" % mae_shear,
             histtype="step", density=True)

    plt.axvline(x=bins_plotting[i], color="k")
    plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlim(10,15)
    plt.xlabel("$\log (M_{\mathrm{predicted}} / \mathrm{M}_{\odot}) $")
    plt.legend(loc="best", fontsize=14)
    plt.subplots_adjust(bottom=0.15)



#Look at histograms of predicted mass for different bins

plt.figure()
mid_bins = (bins_plotting[1:] + bins_plotting[:-1])/2
plt.xlim(10,15)
for i in range(14):
    ids_gbt = (true_z99 >= bins_plotting[i]) & (true_z99 <= bins_plotting[i + 1])
    label = "$\log M \sim %.1f$" % mid_bins[i]
    plt.errorbar(mid_bins[i], np.mean(pred_shear_only[ids_gbt]), yerr=np.std(pred_shear_only[ids_gbt]), fmt="o")
    # plt.hist(pred_shear_only[ids_gbt], bins=20, label=label,
    #          histtype="step", density=True)
    # plt.hist(pred_shear_traj[ids_gbt], bins=50, label="shear+den \n (mae = %.3f" % mae_shear,
    #          histtype="step", density=True)

    # plt.axvline(x=bins_plotting[i], color="k")
    # plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlabel("$\log (M_{\mathrm{predicted}} / \mathrm{M}_{\odot}) $")
    plt.legend(loc="best", fontsize=14)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost//random_sampled_training/shear/shear_only/"
                "predictions_bin_" + str(i) + ".png")
