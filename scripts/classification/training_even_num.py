"""
In this script, I train the algorithm with a training set such that:
- The features are EPS trajectories evaluated at 50 log-spaced mass scales of range 1e10 Msol < M < 1e15 Msol.
- The samples are 100,000 particles s.t. 50,000 are taken from the full set of IN particles ( particles in
    halos 0 to 400) and 50,000 are taken from the full set of OUT particles ( particles in halos 400+
    and particles in no halos), so that one has an even number of IN and OUT particles in the training set.

The algorithm is used to predict probabilities of classes for all remaining particles in the box.
The feature importances of the training set are also store in the `all_out` folder.

"""


import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from mlhalos import machinelearning as ml
from mlhalos import plot
import numpy as np

if __name__ == "__main__":
    # Select 50,000 particles to use as training set

    features_full_mass = np.load("/Users/lls/Documents/CODE/stored_files/all_out/features_full_mass_scale.npy")

    try:
        index_training_in = np.load("/Users/lls/Documents/CODE/stored_files/all_out/even_in_out/50k_features_index_in.npy")
        index_training_out = np.load("/Users/lls/Documents/CODE/stored_files/all_out/even_in_out/50k_features_index_out.npy")
        features_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/even_in_out/50k_features.npy")

    except IOError:
        in_ids = np.load("/Users/lls/Documents/CODE/stored_files/all_out/all_ids_in.npy")
        out_ids = np.load("/Users/lls/Documents/CODE/stored_files/all_out/all_ids_out.npy")

        index_training_in = np.random.choice(len(in_ids), size=50000, replace=False)
        index_training_out = np.random.choice(len(out_ids), size=50000, replace=False)
        index_training_out = index_training_out + len(in_ids)

        features_in = features_full_mass[index_training_in]
        features_out = features_full_mass[index_training_out]
        features_training = np.concatenate((features_in, features_out))

        np.save("/Users/lls/Desktop/try/50k_features_index_in.npy",
                index_training_in)
        np.save("/Users/lls/Desktop/try/50k_features_index_out.npy",
                index_training_out)
        np.save("/Users/lls/Desktop/try/50k_features.npy",
                features_training)

    # Train the algorithm

    algorithm_50k = ml.MLAlgorithm(features_training)

    np.save('/Users/lls/Desktop/try/train_set_x.npy',
            algorithm_50k.X_train)
    np.save('/Users/lls/Desktop/try/test_set_x.npy',
            algorithm_50k.X_test)
    np.save('/Users/lls/Desktop/try/train_set_y.npy',
            algorithm_50k.y_train)
    np.save('/Users/lls/Desktop/try/test_set_y.npy',
            algorithm_50k.y_true)

    # Make predictions on all particles other than the training set

    index_in_out = np.concatenate((index_training_in, index_training_out))
    features_left = features_full_mass[~np.in1d(np.arange(len(features_full_mass)), index_in_out)]

    predicted_probabilities = algorithm_50k.classifier.predict_proba(features_left[:, :-1])
    true_labels = features_left[:,-1]

    np.save('/Users/lls/Desktop/try/predicted_probabilities.npy',
                predicted_probabilities)
    np.save('/Users/lls/Desktop/try/true_labels.npy',
                true_labels)

    # Plot feature importance

    importance = algorithm_50k.classifier.best_estimator_.feature_importances_
    indices = np.argsort(importance)[::-1]

    np.save('/Users/lls/Desktop/try/importance.npy',
            importance)
    np.save('/Users/lls/Desktop/try/indices.npy',
            indices)

    #figure = plot.plot_feature_importance(importance, indices)
    #figure.savefig("/Users/lls/Documents/CODE/stored_files/all_out/even_in_out/feat_importance.png")
