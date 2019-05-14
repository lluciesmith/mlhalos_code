"""
This should be done on hypatia.

"""
import numpy as np
from mlhalos import machinelearning as ml

# load training and test features with EPS label as feature

features_training = np.load("/home/lls/mlhalos_code/stored_files/with_EPS_label/50k_features_w_EPS_label.npy")
features_test = np.load("/home/lls/mlhalos_code/stored_files/with_EPS_label/features_w_EPS_test.npy")

# train algorithm

algo = ml.MLAlgorithm(features_training, split_data_method=None, num_cv_folds=10, n_jobs=22)


# predict probabilities

pred = algo.classifier.predict_proba(features_test[:, :-1])
true = features_test[:, -1]

np.save("/home/lls/mlhalos_code/stored_files/with_EPS_label/predicted_probabilities.npy", pred)
np.save("/home/lls/mlhalos_code/stored_files/with_EPS_label/true_labels.npy", true)


# save classifier details

f_imp = algo.classifier.best_estimator_.feature_importances_
np.save("/home/lls/mlhalos_code/stored_files/with_EPS_label/feature_importances.npy", f_imp)

best_estimator = algo.classifier.best_estimator_
np.savetxt("/home/lls/mlhalos_code/stored_files/with_EPS_label/estimator.txt", best_estimator)

