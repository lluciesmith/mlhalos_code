
"""
Test gradboost on z=0 density features

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from regression.adaboost import gbm_04_only as gbm_fun
import matplotlib.pyplot as plt


saving_path_z0 = "/share/data2/lls/regression/gradboost/randomly_sampled_training/z0_den/nest_2000_lr006/"
features_path = "/share/data2/lls/features_w_periodicity_fix/"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
z0_den_features = np.load(features_path + "z0l_density_contrasts.npy")

ids_in_halo = np.where(halo_mass > 0)[0]
training_set = np.random.choice(ids_in_halo, 250000, replace=False)
testing_set = ids_in_halo[~np.in1d(ids_in_halo, training_set)]
np.save(saving_path_z0 + "training_ids.npy", training_set)
np.save(saving_path_z0 + "testing_ids.npy", testing_set)

z0_den_training = z0_den_features[training_set]
log_halo_training = np.log10(halo_mass[training_set])
training_features = np.column_stack((z0_den_training, log_halo_training))
np.save(saving_path_z0 + "training_features_plus_truth.npy", training_features)

z0_den_testing = z0_den_features[testing_set]
log_halo_testing = np.log10(halo_mass[testing_set])
np.save(saving_path_z0 + "log_halo_testing_set.npy", log_halo_testing)

# training z=0

param_grid = {"loss": "lad", "learning_rate": 0.06, "n_estimators": 2000,  "max_depth": 5,  "max_features":"sqrt"}

clf_z0, pred_test = gbm_fun.train_and_test_gradboost(training_features, z0_den_testing, param_grid=param_grid,
                                                     cv=False)

np.save(saving_path_z0 + "predicted_test_set.npy", pred_test)
joblib.dump(clf_z0, saving_path_z0 + "clf.pkl")



################# GBT VS RF #####################
pred_z0 = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/z0_den"
                  "/nest_2000_lr006/predicted_test_set.npy")
true_z0 = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/z0_den"
                  "/nest_2000_lr006/log_halo_testing_set.npy")

pred_RF_z0 = np.load("/Users/lls/Documents/mlhalos_files/regression/lowz_density/z0/z0_only/predicted_log_halo_mass.npy")
true_RF_z0 = np.log10(np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output"
                              "/even_radii_and_random/true_halo_mass.npy"))


bins_plotting = np.linspace(true_z0.min(), true_z0.max(), 15, endpoint=True)
for i in range(14):
    ids_RF = (true_RF_z0 >= bins_plotting[i]) & (true_RF_z0 <= bins_plotting[i + 1])
    ids_gbt = (true_z0 >= bins_plotting[i]) & (true_z0 <= bins_plotting[i + 1])
    plt.figure()
    plt.hist(pred_RF_z0[ids_RF], bins=50, label="RF", histtype="step", normed=True)
    plt.hist(pred_z0[ids_gbt], bins=50, label="GBT", histtype="step", normed=True)

    plt.axvline(x=bins_plotting[i], color="k")
    plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlim(10,15)
    plt.xlabel("Predicted masses")
    plt.legend(loc="best")
    # plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/z0_den/loss_lab_nest1000_lr001"
    #             "/vs_RF/GBT_vs_RF_bin_" + str(i) + ".png")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/z0_den"
                "/nest_2000_lr006/vs_RF/GBT_vs_RF_bin_" + str(i) + ".png")
