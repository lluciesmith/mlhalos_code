
"""
Get errorbars for importances on z=0 and z=99 training features
from different training sets, randomly-sampled.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from regression.adaboost import gbm_04_only as gbm_fun
from multiprocessing.pool import Pool
from sklearn.ensemble import GradientBoostingRegressor

features_path = "/share/data2/lls/features_w_periodicity_fix/"


def get_random_training_ids(halo_mass):
    ids_in_halos = np.where(halo_mass > 0)[0]
    np.random.seed()
    random_sample = np.random.choice(ids_in_halos, 250000, replace=False)
    return random_sample



################## GET IMPORTANCES FOR z=99 AND z=0 FEATURES #####################

# saving_path_z0 = "/share/data2/lls/regression/gradboost/randomly_sampled_training/z0_den/nest_2000_lr006/"
# saving_path_z99 = "/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
#
# z0_den_features = np.load(features_path + "z0l_density_contrasts.npy")
# traj = np.load(features_path + "ics_density_contrasts.npy")
#
# def train_and_get_imp_GBT(num):
#     print("Loop " + str(num))
#
#     halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
#     tr_ids = get_random_training_ids(halo_mass)
#     print("First training id is " + str(tr_ids[0]))
#
#     training_features_z0 = np.column_stack((z0_den_features[tr_ids], np.log10(halo_mass[tr_ids])))
#     training_features_z99 = np.column_stack((traj[tr_ids], np.log10(halo_mass[tr_ids])))
#
#     param_grid = {"loss": "lad",  "learning_rate": 0.06,  "n_estimators": 2000, "max_depth": 5, "max_features": 15,
#                   "criterion":"mse"}
#     clf_z0 = gbm_fun.train_gbm(training_features_z0, param_grid=param_grid, cv=False, save=False)
#     imp_z0 = clf_z0.feature_importances_
#     np.save(saving_path_z0 + "imp_" + str(num) + ".npy", imp_z0)
#
#     clf_z99 = gbm_fun.train_gbm(training_features_z99, param_grid=param_grid, cv=False, save=False)
#     imp_z99 = clf_z99.feature_importances_
#     np.save(saving_path_z99 + "imp_" + str(num) + ".npy", imp_z99)
#
#     return imp_z0, imp_z99
#
# pool = Pool(processes=24)
# imps = pool.map(train_and_get_imp_GBT, np.arange(24))
# pool.close()
# pool.join()
#
# imps_0 = np.array([imps[i][0] for i in range(24)])
# imps_99 = np.array([imps[i][1] for i in range(24)])
#
# np.save(saving_path_z0 + "all_imps_z0.npy", imps_0)
# np.save(saving_path_z99 + "all_imps_z99.npy", imps_99)


################## GET IMPORTANCE FOR SHEAR + TRAJECTORIES #####################

saving_path_shear_plus_traj = "/share/data2/lls/regression/gradboost/randomly_sampled_training/shear/shear_and_ic_traj/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")
den_sub_ell = np.lib.format.open_memmap(features_path + "density_subtracted_ellipticity.npy", mode="r",
                                        shape=(256**3, 50))
den_sub_prol = np.lib.format.open_memmap(features_path + "density_subtracted_prolateness.npy", mode="r",
                                         shape=(256**3, 50))

def train_and_get_imp_GBT_for_shear(num):
    print("Loop " + str(num))

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    tr_ids = get_random_training_ids(halo_mass)
    print("First training id is " + str(tr_ids[0]))

    training_features = np.column_stack((traj[tr_ids], den_sub_ell[tr_ids], den_sub_prol[tr_ids],
                                         np.log10(halo_mass[tr_ids])))

    param_grid = {"loss": "lad",  "learning_rate": 0.06,  "n_estimators": 2000, "max_depth": 5, "max_features": "sqrt",
                  "criterion":"mse"}
    clf_shear = gbm_fun.train_gbm(training_features, param_grid=param_grid, cv=False, save=False)
    imp_shear = clf_shear.feature_importances_
    np.save(saving_path_shear_plus_traj + "imp_" + str(num) + ".npy", imp_shear)
    return imp_shear


pool = Pool(processes=40)
imps = pool.map(train_and_get_imp_GBT_for_shear, np.arange(40))
pool.close()
pool.join()

imps_shear = np.array([imps[i] for i in range(40)])

np.save(saving_path_shear_plus_traj + "all_imps_shear.npy", imps_shear)
