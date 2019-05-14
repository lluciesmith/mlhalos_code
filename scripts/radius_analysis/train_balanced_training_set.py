import numpy as np
import sys
sys.path.append('/home/lls/mlhalos_code')
from mlhalos import parameters
from mlhalos import machinelearning as ml

ic = parameters.InitialConditionsParameters(load_final=True)

ids_IN = ic.ids_IN
ids_OUT = ic.ids_OUT

# split in ids in radial ranges

r_in = np.load("/home/lls/stored_files/radii_stuff/radii_properties_in_ids.npy")

in_inner = r_in[:,0][r_in[:, 2] <= 0.3]
in_mid = r_in[:,0][(r_in[:, 2] > 0.3) & (r_in[:, 2] <= 0.6)]
in_outer = r_in[:,0][(r_in[:, 2] > 0.6) & (r_in[:, 2] <= 1)]
in_rest = r_in[:,0][r_in[:, 2] > 1]

# split out ids in radial ranges
# r_out = np.load("/Users/lls/Documents/CODE/stored_files/radii_stuff/radii_properties_out_ids.npy")

# split in training and testing

training_index = np.random.choice(np.arange(len(in_rest)), 5000, replace=False)
training_ids_IN = np.concatenate((in_inner[training_index], in_mid[training_index],
                                  in_outer[training_index], in_rest[training_index]))
training_index_out = np.random.choice(np.arange(len(ic.ids_OUT)), 30000, replace=False)
training_ids_OUT = ic.ids_OUT[training_index_out]

ids_in_out = np.concatenate((ic.ids_IN, ic.ids_OUT))
training_ids = np.concatenate((training_ids_IN, training_ids_OUT))

training_indices = np.in1d(ids_in_out, training_ids)
testing_indices = ~np.in1d(ids_in_out, training_ids)
assert np.allclose(np.sort(ids_in_out[training_indices]), np.sort(training_ids))

path = "/share/data1/lls/shear_quantities/"
den_features = np.load(path + "density_features.npy")
training_features = den_features[training_indices]
testing_features = den_features[testing_indices]
del den_features

np.save("/home/lls/stored_files/radii_stuff/balance_training_set/training_features_indices.npy", training_indices)
np.save("/home/lls/stored_files/radii_stuff/balance_training_set/testing_features_indices.npy", testing_indices)
np.save("/home/lls/stored_files/radii_stuff/balance_training_set/training_features.npy", training_features)
np.save("/home/lls/stored_files/radii_stuff/balance_training_set/testing_features.npy", testing_features)
del testing_features

clf = ml.MLAlgorithm(training_features, algorithm='Random Forest', split_data_method=None, cross_validation=False,
                     num_cv_folds=5, tree_ids=False, n_jobs=60, save=False)

# print(clf.best_estimator)
del training_features

# joblib.dump(clf, path + "classifier_den/clf_upgraded.pkl", compress=3)

testing_features = np.load("/home/lls/stored_files/radii_stuff/balance_training_set/testing_features.npy")
y_predicted = clf.classifier.predict_proba(testing_features[:, :-1])
y_true = testing_features[:, -1]

np.save("/home/lls/stored_files/radii_stuff/balance_training_set/predicted_den.npy", y_predicted)
np.save("/home/lls/stored_files/radii_stuff/balance_training_set/true_den.npy", y_true)
