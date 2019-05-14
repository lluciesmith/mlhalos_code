import numpy as np
#import sys
#sys.path.append('/Users/lls/Documents/mlhalos_code')
from mlhalos import machinelearning as ml

# load training and test features with EPS label as feature

features_training = np.load("/home/lls/mlhalos_code/stored_files/50k_features.npy")
features_training = np.column_stack((features_training[:, :-1], features_training[:, -1], features_training[:, -1]))

features_test = np.load("/home/lls/mlhalos_code/stored_files/features_test.npy")
features_test = np.column_stack((features_test[:, :-1], features_test[:, -1], features_test[:, -1]))

# train algorithm

algo = ml.MLAlgorithm(features_training, split_data_method=None, cross_validation=False, num_cv_folds=10, n_jobs=22)

# predict probabilities

pred = algo.classifier.predict(features_test[:, :-1])
true = features_test[:, -1]

np.save("/home/lls/mlhalos_code/stored_files/true_label_feature/predicted_probabilities.npy", pred)
np.save("/home/lls/mlhalos_code/stored_files/true_label_feature/true_labels.npy", true)