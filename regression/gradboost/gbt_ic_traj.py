
"""
Test gradboost on z=99 density features

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from regression.adaboost import gbm_04_only as gbm_fun
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

saving_path_traj = "/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
features_path = "/share/data2/lls/features_w_periodicity_fix/"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

ids_in_halo = np.where(halo_mass > 0)[0]
training_set = np.random.choice(ids_in_halo, 250000, replace=False)
np.save(saving_path_traj + "training_ids.npy", training_set)
testing_set = ids_in_halo[~np.in1d(ids_in_halo, training_set)]
np.save(saving_path_traj + "testing_ids.npy", testing_set)


traj_training = traj[training_set]
log_halo_training = np.log10(halo_mass[training_set])
training_features = np.column_stack((traj_training, log_halo_training))
np.save(saving_path_traj + "training_features_plus_truth.npy", training_features)

traj_testing = traj[testing_set]
log_halo_testing = np.log10(halo_mass[testing_set])
np.save(saving_path_traj + "log_halo_testing_set.npy", log_halo_testing)

# training z=0

param_grid = {"loss": "lad", "learning_rate": 0.06, "n_estimators": 2000,  "max_depth": 5,  "max_features":"sqrt"}

clf_z0, pred_test = gbm_fun.train_and_test_gradboost(training_features, traj_testing, param_grid=param_grid,
                                                     cv=False)

np.save(saving_path_traj + "predicted_test_set.npy", pred_test)
joblib.dump(clf_z0, saving_path_traj + "clf.pkl")


#################### z=99 vs z=0 GBT #################################

pred_z99 = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj"
                   "/nest_2000_lr006/predicted_test_set.npy")
true_z99 = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj"
                   "/nest_2000_lr006/log_halo_testing_set.npy")
pred_z0 = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/z0_den"
                  "/nest_2000_lr006/predicted_test_set.npy")
true_z0 = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/z0_den"
                  "/nest_2000_lr006/log_halo_testing_set.npy")

bins_plotting = np.linspace(true_z0.min(), true_z0.max(), 15, endpoint=True)
for i in range(14):
    ids_bin_z0 = (true_z0 >= bins_plotting[i]) & (true_z0 <= bins_plotting[i + 1])
    mae_z0= mean_absolute_error(true_z0[ids_bin_z0], pred_z0[ids_bin_z0])

    ids_bin_z99 = (true_z99 >= bins_plotting[i]) & (true_z99 <= bins_plotting[i + 1])
    mae_z99 = mean_absolute_error(true_z99[ids_bin_z99], pred_z99[ids_bin_z99])

    plt.figure()
    plt.hist(pred_z0[ids_bin_z0], bins=50, label="z=0, mae = %.3f" % mae_z0, color="g", histtype="step",
             normed=True)
    plt.hist(pred_z99[ids_bin_z99], bins=50, label="z=99, mae = %.3f" % mae_z99, color="b", histtype="step",
             normed=True)

    plt.axvline(x=bins_plotting[i], color="k")
    plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlim(10,15)
    plt.xlabel("Predicted masses")
    plt.legend(loc="best")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj"
                "/nest_2000_lr006/vs_z=0/bin_" + str(i) + "_z_99_vs_z_0.png")


###################### z=99 GBT vs RF ####################################

pred_RF_z99 = np.log10(np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output"
                              "/even_radii_and_random/predicted_halo_mass.npy"))
true_RF_z99 = np.log10(np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output"
                              "/even_radii_and_random/true_halo_mass.npy"))
for i in range(14):
    ids_RF = (true_RF_z99 >= bins_plotting[i]) & (true_RF_z99 <= bins_plotting[i + 1])
    ids_gbt = (true_z99 >= bins_plotting[i]) & (true_z99 <= bins_plotting[i + 1])
    plt.figure()
    plt.hist(pred_RF_z99[ids_RF], bins=50, label="RF", histtype="step", normed=True)
    plt.hist(pred_z99[ids_gbt], bins=50, label="GBT", histtype="step", normed=True)

    plt.axvline(x=bins_plotting[i], color="k")
    plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlim(10,15)
    plt.xlabel("Predicted masses")
    plt.legend(loc="best")
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj"
                "/nest_2000_lr006/vs_RF/GBT_vs_RF_bin_" + str(i) + ".png")