# Choose the IN/OUT boundary to be at halo 4000 which is ~1.4 x 10^11 Msol
# Everything is saved in `/share/data1/lls/h4000_in_out_boundary`

# N.B. Don't bother re-ordering the features to have first all in and then
# all out particles. Therefore training indices are identical to traning particles

save_dir = "/share/data1/lls/h74_in_out_boundary/"

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import machinelearning as ml


ic = parameters.InitialConditionsParameters(load_final=True, max_halo_number=74)
num_ids = len(ic.final_snapshot)

true_labels = np.ones((num_ids,))
true_labels[ic.ids_OUT] *= -1

try:
    training_index = np.load(save_dir + "training_index.npy")
    testing_index = np.load(save_dir + "testing_index.npy")
except:
    all_ids = np.arange(len(ic.final_snapshot),)
    training_index = np.random.choice(all_ids, 50000)
    np.save(save_dir + "training_index.npy", training_index)
    testing_index = all_ids[~np.in1d(all_ids, training_index)]
    np.save(save_dir + "testing_index.npy", testing_index)

# Train based on density field only

# den_features = np.load(path + "density_features.npy")
den_features = np.lib.format.open_memmap("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy",
                                         mode="r", shape=(num_ids, 50))

den_training = den_features[training_index]
features_training = np.column_stack((den_training, true_labels[training_index]))
del den_training

RF = ml.MLAlgorithm(features_training, split_data_method=None, n_jobs=60, save=True,
                    path=save_dir + "den_classifier/classifier.pkl")
clf = RF.algorithm
print(RF.best_estimator)
np.save(save_dir + "density/feature_importances.npy", RF.feature_importances)

pred_prob = RF.algorithm.predict_proba(den_features[testing_index])
np.save(save_dir + "density/predicted_probabilities.npy", pred_prob)
true_test_label = true_labels[testing_index]
np.save(save_dir + "density/true_test_labels.npy", true_test_label)