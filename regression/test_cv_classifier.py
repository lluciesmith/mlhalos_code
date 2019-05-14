"""
Train regression classifier and save

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")
true_labels = np.ones((256**3))
true_labels[np.random.choice(range(256**3), int(256**3/3), replace=False)] = -1

try:
    training_ind = np.load("/share/data1/lls/regression/50k_training_ids.npy")
    testing_ind = np.load("/share/data1/lls/regression/50k_testing_ids.npy")
except IOError:
    print("Generating training/testing indices and saving")
    training_ind = np.random.choice(len(traj), 50000)
    np.save("/share/data1/lls/try_classifier/50k_training_ids.npy", training_ind)
    testing_ind = np.arange(len(traj))[~np.in1d(range(len(traj)), training_ind)]
    np.save("/share/data1/lls/try_classifier/50k_testing_ids.npy", training_ind)

feat_training = np.column_stack((traj[training_ind], true_labels[training_ind]))
# X_test = traj[testing_ind]
# y_test = halo_mass[testing_ind]
del traj
del true_labels

clf = ml.MLAlgorithm(feat_training, method="classification", split_data_method=None, n_jobs=60, save=True,
                     path="/share/data1/lls/regression/try_classifier/classifier.pkl")

print(clf.best_estimator)
print(clf.algorithm.best_params_)
print(clf.algorithm.best_score_)
np.save("/share/data1/lls/regression/try_classifier/f_imp.npy", clf.feature_importances)

# y_predicted = clf.algorithm.predict(X_test)
# np.save("/share/data1/lls/regression/predicted_halo_mass.npy", y_predicted)
# np.save("/share/data1/lls/regression/true_halo_mass.npy", y_test)
