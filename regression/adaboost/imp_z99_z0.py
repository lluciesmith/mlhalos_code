
"""
Get errorbars for importances on z=0 and z=99 training features
from different training sets.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from regression.adaboost import gbm_04_only as gbm_fun
from multiprocessing.pool import Pool

saving_path_z0 = "/share/data2/lls/regression/gradboost/z0_den/imp_nest600/"
saving_path_z99 = "/share/data2/lls/regression/gradboost/ic_traj/imp_nest600/"
features_path = "/share/data2/lls/features_w_periodicity_fix/"


def get_random_training_ids(halo_mass):
    radii_path = "/home/lls/stored_files/radii_stuff/"
    halo_mass_in_ids = halo_mass[halo_mass > 0]

    # sort ids in halos and corresponding r/r_vir value

    radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
    radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")
    fraction = np.concatenate((radii_properties_in[:, 2], radii_properties_out[:, 2]))
    ids_in_halo = np.concatenate((radii_properties_in[:, 0], radii_properties_out[:, 0]))
    ind_sorted = np.argsort(ids_in_halo)

    ids_in_halo_mass = ids_in_halo[ind_sorted].astype("int")
    r_fraction = fraction[ind_sorted]
    del fraction
    del ids_in_halo

    # Select a balanced training set
    # Take particle ids in each halo mass bin

    n, log_bins = np.histogram(np.log10(halo_mass_in_ids), bins=50)
    bins = 10 ** log_bins

    training_ind = []
    for i in range(len(bins) - 1):
        ind_bin = np.where((halo_mass_in_ids >= bins[i]) & (halo_mass_in_ids < bins[i + 1]))[0]
        ids_in_mass_bin = ids_in_halo_mass[ind_bin]

        if ids_in_mass_bin.size == 0:
            print("Pass")
            pass

        else:
            if i == 49:
                num_p = 2000
            else:
                num_p = 1000

            radii_in_mass_bin = r_fraction[ind_bin]

            np.random.seed()

            ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
            ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
                                      replace=False)
            ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
                                     replace=False)
            ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)

            training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
            training_ind.append(training_ids_in_bin)

    training_ind = np.concatenate(training_ind)

    remaining_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ind)]
    np.random.seed()
    random_sample = np.random.choice(remaining_ids, 50000, replace=False)

    training_ind = np.concatenate((training_ind, random_sample))
    return training_ind

# data

z0_den_features = np.load(features_path + "z0l_density_contrasts.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")


def train_and_get_imp_GBT(num):
    print("Loop " + str(num))

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    tr_ids = get_random_training_ids(halo_mass)

    training_features_z0 = np.column_stack((z0_den_features[tr_ids], np.log10(halo_mass[tr_ids])))
    training_features_z99 = np.column_stack((traj[tr_ids], np.log10(halo_mass[tr_ids])))

    param_grid = {"loss": "lad",  "learning_rate": 0.01,  "n_estimators": 600, "max_depth": 5, "max_features": 15}
    clf_z0 = gbm_fun.train_gbm(training_features_z0, param_grid=param_grid, cv=False, save=False)
    imp_z0 = clf_z0.feature_importances_
    np.save(saving_path_z0 + "imp_" + str(num) + ".npy", imp_z0)

    clf_z99 = gbm_fun.train_gbm(training_features_z99, param_grid=param_grid, cv=False, save=False)
    imp_z99 = clf_z99.feature_importances_
    np.save(saving_path_z99 + "imp_" + str(num) + ".npy", imp_z99)

    return imp_z0, imp_z99

pool = Pool(processes=12)
imps = pool.map(train_and_get_imp_GBT, np.arange(12))
pool.close()
pool.join()

imps_0 = np.array([imps[i][0] for i in range(12)])
imps_99 = np.array([imps[i][1] for i in range(12)])

np.save(saving_path_z0 + "all_imps_z0.npy", imps_0)
np.save(saving_path_z0 + "all_imps_z99.npy", imps_99)