"""
In this script, I train the algorithm with a training set such that:
- The features are EPS trajectories evaluated at 50 log-spaced mass scales of range 1e10 Msol < M < 1e15 Msol.
- The samples are are 50,000 random particles taken from the full set of IN particles ( particles in halos 0 to 400)
    and OUT particles ( particles in halos 400+ and particles in no halos.)

The algorithm is used to predict probabilities of classes for all remaining particles in the box.
The feature importances of the training set are also store in the `all_out` folder.

"""


import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from mlhalos import machinelearning as ml
from mlhalos import plot
import numpy as np


# Select 50,000 particles to use as training set


def load_features(features_type="test",
                  all_features="/Users/lls/Documents/CODE/stored_files/all_out/features_full_mass_scale.npy",
                  index_training_features="/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy"):
    features_full_mass = np.load(all_features)

    if features_type == "all":
        return features_full_mass

    elif features_type == "training":
        index_training = np.load(index_training_features)
        features_training = features_full_mass[index_training]
        return features_training

    elif features_type == "test":
        index_training = np.load(index_training_features)
        features_left = features_full_mass[~np.in1d(np.arange(len(features_full_mass)), index_training)]
        return features_left

    else:
        NameError("Not a valid features_type")


def train_algorithm(features):
    trained_algorithm = ml.MLAlgorithm(features)
    return trained_algorithm


def find_predicted_and_true_labels(trained_algorithm, features):
    predicted_probabilities = trained_algorithm.algorithm.predict_proba(features[:, :-1])
    true_labels = features[:, -1]
    return predicted_probabilities, true_labels


######################## SCRIPT ########################


if __name__ == "__main__":

    # Train the algorithm

    features_training = load_features(features_type="training")
    trained_classifier = ml.MLAlgorithm(features_training)

    # Make predictions on all particles other than the training set

    test_features = load_features(features_type="test")
    predicted_probabilities, true_labels = find_predicted_and_true_labels(trained_classifier, test_features)

    np.save('/Users/lls/Documents/CODE/stored_files/all_out/predicted_probabilities.npy',
            predicted_probabilities)
    np.save('/Users/lls/Documents/CODE/stored_files/all_out/true_labels.npy',
            true_labels)


# # Plot feature importance
#
# importance = algorithm_50k.algorithm.best_estimator_.feature_importances_
# indices = np.argsort(importance)[::-1]
#
# np.save('/Users/lls/Documents/CODE/stored_files/all_out/importance.npy',
#         importance)
# np.save('/Users/lls/Documents/CODE/stored_files/all_out/indices.npy',
#         indices)
#
# figure = plot.plot_feature_importance(importance, indices)
# figure.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feat_importance.png")

