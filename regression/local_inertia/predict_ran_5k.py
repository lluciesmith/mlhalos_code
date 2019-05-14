import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
import os
import re
from sklearn.externals import joblib


def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)


def merge_test_set(path, particle_ids, save=True):
    eig = np.zeros((len(particle_ids), 50, 3))
    for i in range(len(particle_ids)):
        eig[i] = np.load(path + "random/eigenvalues_particle_" + str(particle_ids[i]) + ".npy")
    return eig

if __name__ == "__main__":

    path = "/share/data2/lls/regression/local_inertia/"
    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

    ids_test_set = np.load("/share/data2/lls/regression/local_inertia/tensor/ran_5k.npy")
    ids_test_set = np.sort(ids_test_set)
    eig_test_set = merge_test_set(path + "tensor/", ids_test_set, save=True)

    training_ids = np.load(path + "reduced_training_set.npy")
    indices_no_training = ~np.in1d(ids_test_set, training_ids)

    ids_test = ids_test_set[indices_no_training]
    #np.save(path + "ran_5k/ids_tested.npy", ids_test)
    #np.save(path + "ran_5k/true_test_halo_mass.npy", halo_mass[ids_test])

    ids_test_check = np.load(path + "ran_5k/ids_tested.npy")
    assert np.allclose(ids_test, ids_test_check)


    # features - local inertia eigenvalues

    eig0_test = eig_test_set[indices_no_training, :, 0]
    eig1_test = eig_test_set[indices_no_training, :, 1]
    eig2_test = eig_test_set[indices_no_training, :, 2]

    # features - density

    # path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
    # den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))
    # den_testing = den_features[ids_test]

    # Predict with classifier trained on density + local inertia

    # X_test = np.column_stack((den_testing, eig0_test, eig1_test, eig2_test))
    X_test = np.column_stack((eig0_test, eig1_test, eig2_test))
    clf_inertia = joblib.load('/share/data2/lls/regression/local_inertia/inertia_only/classifier/classifier.pkl')
    y_predicted = clf_inertia.predict(X_test)
    np.save(path + "inertia_only/5k_predicted_halo_mass.npy", 10**y_predicted)

    # # Predict with classifier trained on density + local inertia
    #
    # clf_den = joblib.load('/share/data2/lls/regression/local_inertia/inertia_only/clf_den_only/classifier.pkl')
    # den_y_predicted = clf_den.predict(den_testing)
    # np.save(path + "ran_5k/clf_den_only/predicted_halo_mass.npy", 10 ** den_y_predicted)