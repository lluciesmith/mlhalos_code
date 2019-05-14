import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from sklearn.externals import joblib

# Get training set

saving_path = "/share/data2/lls/multiclass/inertia_plus_den/"

# # sort ids in halos and corresponding r/r_vir value
radii_path = "/home/lls/stored_files/radii_stuff/"
radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")
fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
ind_sorted = np.argsort(ids_in_halo)

ids_in_halo_mass = ids_in_halo[ind_sorted].astype("int")
r_fraction = fraction[ind_sorted]
del fraction
del ids_in_halo

# training set

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
assert np.allclose(ids_in_halo_mass, np.where(halo_mass != 0)[0])
log_halo_mass_in_ids = np.log10(halo_mass[ids_in_halo_mass])

bins = np.concatenate((np.linspace(log_halo_mass_in_ids.min(), 14, 15), [log_halo_mass_in_ids.max() + 0.1]))
n, bins = np.histogram(log_halo_mass_in_ids, bins=bins)

training_ind = []
class_labels = np.ones(len(halo_mass)) * (-1)

for i in range(len(bins) - 1):
    ind_bin = np.where((log_halo_mass_in_ids >= bins[i]) & (log_halo_mass_in_ids < bins[i + 1]))[0]
    ids_in_mass_bin = ids_in_halo_mass[ind_bin]

    if ids_in_mass_bin.size == 0:
        print("Pass")
        pass

    else:
        class_labels[ids_in_mass_bin] = i
        num_p = 5000

        radii_in_mass_bin = r_fraction[ind_bin]

        ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
        print((num_p/len(ids_in_mass_bin[radii_in_mass_bin < 0.3]) * 100))
        ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
                                  replace=False)
        print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)]) * 100))
        ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
                                 replace=False)
        print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)]) * 100))
        ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)
        print((num_p / len(ids_in_mass_bin[radii_in_mass_bin >= 1]) * 100))

        training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
        training_ind.append(training_ids_in_bin)

training_ind = np.concatenate(training_ind)
np.save(saving_path + "training_ids.npy", training_ind)

assert np.allclose(np.where(class_labels== -1)[0], np.where(halo_mass == 0)[0])

testing_ids = ids_in_halo_mass[~np.in1d(ids_in_halo_mass, training_ind)]
np.save(saving_path + "testing_ids.npy", testing_ids)

y_test = class_labels[testing_ids]
np.save(saving_path + "classes_test_set.npy", y_test)


##### TRAINING + PREDICTING #######

path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))

path_inertia = "/share/data1/lls/regression/inertia/cores_40/"
eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))

feat_training = np.column_stack((den_features[training_ind], eig_0[training_ind], class_labels[training_ind]))
# X_test = np.column_stack((den_features[testing_ids], eig_0[testing_ids]))

cv = True
param_grid = {"n_estimators": [800, 1000, 1300],
              "max_features": ["auto", 0.4],
              "min_samples_leaf": [15, 5],
              "criterion": ["gini", "entropy"]
              }

clf = ml.MLAlgorithm(feat_training, method="classification", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, param_grid=param_grid,
                     path=saving_path + "classifier/classifier.pkl")
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)
np.save(saving_path + "f_imp.npy", clf.feature_importances)

# classify

clf = joblib.load(saving_path + "classifier/classifier.pkl")
testing_ids = np.load(saving_path + "testing_ids.npy")
X_test = np.column_stack((den_features[testing_ids], eig_0[testing_ids]))

y_predicted = clf.algorithm.predict_proba(X_test)
np.save(saving_path + "predicted_classes.npy", y_predicted)

# predict probabilities in chuncks

a = np.array_split(range(len(testing_ids)), 100)

for i in range(100):
    t = X_test[a[i], :]
    pred = clf.algorithm.predict_proba(t)
    np.save(saving_path + "pred/predicted_" + str(a[i][-1]) + ".npy", pred)

p = np.load(saving_path + "pred/predicted_" + str(a[0][-1]) + ".npy")
for i in range(1, 100):
    p = np.vstack((p, np.load(saving_path + "pred/predicted_" + str(a[i][-1]) + ".npy")))
np.save(saving_path + "predicted_classes.npy", p)


