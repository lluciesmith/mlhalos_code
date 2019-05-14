import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml


def reduce_training_set(training_particles, halo_mass_all_particles, bins, num_ids_per_bin=None):
    h_m_training = np.log10(halo_mass_all_particles[training_particles])
    n, b = np.histogram(h_m_training, bins=bins)

    if num_ids_per_bin is None:
        num_ids_per_bin = n[n != 0].min()

    training_ind = []
    for i in range(len(bins) - 1):
        ind_bin = np.where((h_m_training >= bins[i]) & (h_m_training < bins[i + 1]))[0]

        if ind_bin.size != 0:
            if i == 49:
                ind_bin = np.where((h_m_training >= bins[i]) & (h_m_training <= bins[i + 1]))[0]
                num_ids_per_bin = n[-1]

            ids_in_mass_bin = training_particles[ind_bin]
            if len(ids_in_mass_bin) < num_ids_per_bin:
                ran = ids_in_mass_bin
            else:
                ran = np.random.choice(ids_in_mass_bin, num_ids_per_bin, replace=False)
            training_ind.append(ran)
    training_ind = np.concatenate(training_ind)
    return training_ind


if __name__ == "__main__":
    # Get training set

    saving_path = "/share/data2/lls/regression/local_inertia/"
    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

    training_ids = np.load("/share/data2/lls/regression/local_inertia/tensor/first_try/training_particles_saved.npy")
    # reduced_ids = np.load(saving_path + "reduced_training_set.npy")

    n_tot, b_tot = np.histogram(np.log10(halo_mass[halo_mass!=0]), bins=50)
    reduced_ids = reduce_training_set(training_ids, halo_mass, b_tot, num_ids_per_bin=3000)
    reduced_ids = np.sort(reduced_ids)
    np.save(saving_path + "reduced_training_set.npy", reduced_ids)
    print("The number of training particles is " + str(len(training_ids)))
    assert np.allclose(np.sort(training_ids), training_ids)
    assert np.allclose(np.sort(reduced_ids), reduced_ids)

    indices_reduced = np.in1d(training_ids, reduced_ids)

    eig_training = np.load("/share/data2/lls/regression/local_inertia/tensor/first_try/training_eigenvalues_particles.npy")
    eig_reduced = eig_training[indices_reduced]

    eig0 = eig_reduced[:, :, 0]
    eig1 = eig_reduced[:, :, 1]
    eig2 = eig_reduced[:, :, 2]

    path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
    den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
    den_training = den_features[reduced_ids]

    log_mass = np.log10(halo_mass[reduced_ids])

    # train

    training_features = np.column_stack((den_training, eig0, eig1, eig2, log_mass))
    print(training_features.shape)

    cv = True
    third_features = int((training_features.shape[1] - 1)/3)
    half_features = int((training_features.shape[1] - 1)/2)
    quarter_features = int((training_features.shape[1] - 1)/4)
    param_grid = {"n_estimators": [1000, 1300, 1600],
                  "max_features": [third_features, quarter_features, half_features],
                  "min_samples_split": [2],
                  "min_samples_leaf": [1]
                  #"criterion": ["mse", "mae"],
                  }

    clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                         save=True, path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
    if cv is True:
        print(clf.best_estimator)
        print(clf.algorithm.best_params_)
        print(clf.algorithm.best_score_)

    np.save(saving_path + "f_imp.npy", clf.feature_importances)
