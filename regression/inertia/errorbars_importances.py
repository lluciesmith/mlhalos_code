import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# Get errorbars for feature importances for case of training on inertia tensor only


def get_random_sample_training_ids(particle_ids, halo_mass_particles, radial_fraction_particles, mass_bins):
    training_ind = []

    for i in range(len(mass_bins) - 1):
        ind_bin = np.where((halo_mass_particles >= mass_bins[i]) & (halo_mass_particles < mass_bins[i + 1]))[0]
        ids_in_mass_bin = particle_ids[ind_bin]

        if ids_in_mass_bin.size == 0:
            print("Pass")
            pass

        else:
            if i == 49:
                num_p = 2000
            else:
                num_p = 1000

            radii_in_mass_bin = radial_fraction_particles[ind_bin]

            ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
            ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
                                      replace=False)
            ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
                                     replace=False)
            ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)

            training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
            training_ind.append(training_ids_in_bin)

    training_ind = np.concatenate(training_ind)

    remaining_ids = particle_ids[~np.in1d(particle_ids, training_ind)]
    random_sample = np.random.choice(remaining_ids, 50000, replace=False)

    training_ind = np.concatenate((training_ind, random_sample))
    return training_ind


if __name__ == "__main__":
    saving_path = "/share/data1/lls/regression/inertia/"

    path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
    den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256 ** 3, 50))

    path_inertia = "/share/data1/lls/regression/inertia/cores_40/"
    eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))
    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

    # Load stuff to construct training set

    radii_path = "/home/lls/stored_files/radii_stuff/"
    radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
    radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")
    fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
    ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
    ind_sorted = np.argsort(ids_in_halo)

    ids_in_halo_mass = ids_in_halo[ind_sorted].astype("int")
    r_fraction = fraction[ind_sorted]

    halo_mass_in_ids = halo_mass[halo_mass>0]
    n, log_bins = np.histogram(np.log10(halo_mass_in_ids), bins=50)
    bins = 10**log_bins

    f_imp_all = np.zeros((10, 100))
    for i in range(10):
        training_ids = get_random_sample_training_ids(ids_in_halo_mass, halo_mass_in_ids, r_fraction, bins)

        eig_0_training = eig_0[training_ids]
        den_training = den_features[training_ids]
        log_mass = np.log10(halo_mass[training_ids])

        training_features = np.column_stack((den_training, eig_0_training, log_mass))
        print(training_features.shape)

        cv = True
        third_features = int((training_features.shape[1] - 1) / 3)
        param_grid = {"n_estimators": [1000, 1300],
                      "max_features": [third_features, "sqrt", 5, 10],
                      "min_samples_leaf": [5, 15],
                      # "criterion": ["mse", "mae"],
                      }

        clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None,
                             n_jobs=60, param_grid=param_grid)
        if cv is True:
            print(clf.best_estimator)
            print(clf.algorithm.best_params_)
            print(clf.algorithm.best_score_)

        np.save(saving_path + "f_imp_" + str(i) + ".npy", clf.feature_importances)
        f_imp_all[i] = clf.feature_importances

    np.save(saving_path + "f_imp_all.npy", f_imp_all)

