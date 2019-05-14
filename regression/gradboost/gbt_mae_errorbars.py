"""
In order to know if shear is improving ic traj feature set then we must place errorbars on the
MAE at each mass bin

"""


from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error



def get_mae_each_mass_bin(bins, true, predicted):
    mae_bins = np.zeros((len(bins) - 1, ))
    for i in range(len(bins) - 1):
        ids_bin = (true >= bins[i]) & (true <= bins[i + 1])
        mae_bins[i] = mean_absolute_error(true[ids_bin], predicted[ids_bin])
    return mae_bins


def get_mae_random_test_set(halo_mass_test_set, testing_features, classifier):
    np.random.seed()
    ran_test_set = np.random.choice(np.arange(len(halo_mass_test_set)), 500000, replace=False)

    true_random = halo_mass_test_set[ran_test_set]
    pred_random_test = classifier.predict(testing_features[ran_test_set])

    mae_random_test = mean_absolute_error(true_random, pred_random_test)

    bins_plotting = np.linspace(halo_mass_test_set.min(), halo_mass_test_set.max(), 15, endpoint=True)
    mae_bins_random = get_mae_each_mass_bin(bins_plotting, true_random, pred_random_test)
    return mae_random_test, mae_bins_random


ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/testing_ids.npy")

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_halo_testing = np.log10(halo_mass[ids])


############# SHEAR MAE ERRORBAR ##############

#
# f_testing = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/shear/shear_and_ic_traj/"
#                     "testing_features.npy")
#
# saving_path_shear_plus_traj = "/share/data2/lls/regression/gradboost/randomly_sampled_training/shear/shear_and_ic_traj/"
# clf_shear = joblib.load(saving_path_shear_plus_traj + "clf.pkl")
#
# mae_all = np.zeros((100, ))
# mae_per_bins = np.zeros((100, 14))
# for i in range(100):
#     mae_all[i], mae_per_bins[i] = get_mae_random_test_set(log_halo_testing, f_testing, clf_shear)
#
#
# np.save(saving_path_shear_plus_traj + "mae_all.npy", mae_all)
# np.save(saving_path_shear_plus_traj + "mae_bins.npy", mae_per_bins)



######### IC TRAJECTORIES MAE ERRORBAR #######

traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")
traj_testing = traj[ids]

saving_path_traj = "/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
clf_traj = joblib.load(saving_path_traj + "clf.pkl")

mae_all = np.zeros((100, ))
mae_per_bins = np.zeros((100, 14))
for i in range(100):
    mae_all[i], mae_per_bins[i] = get_mae_random_test_set(log_halo_testing, traj_testing, clf_traj)


np.save(saving_path_traj + "mae_all.npy", mae_all)
np.save(saving_path_traj + "mae_bins.npy", mae_per_bins)
