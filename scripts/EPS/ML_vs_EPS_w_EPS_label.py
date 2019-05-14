"""
In this script, I train the algorithm with a training set such that:
- The features are EPS trajectories evaluated at 50 log-spaced mass scales of range 1e10 Msol < M < 1e15 Msol.
- The samples are 100,000 particles s.t. 50,000 are taken from the full set of IN particles ( particles in
    halos 0 to 400) and 50,000 are taken from the full set of OUT particles ( particles in halos 400+
    and particles in no halos), so that one has an even number of IN and OUT particles in the training set.

The algorithm is used to predict probabilities of classes for all remaining particles in the box.
The feature importances of the training set are also store in the `all_out` folder.

"""

# FOR HYPATIA - CORES24:

import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from mlhalos import machinelearning as ml
from mlhalos import parameters
from utils import classification_results

import numpy as np

if __name__ == "__main__":

        all_features_with_EPS = np.load('/home/lls/stored_files/with_EPS_label/features_all_full_mass.npy')

        ############### ML PREDICTIONS ################

        # Pick randomly 50,000 for training

        index_training = np.random.choice(len(all_features_with_EPS), size=50000, replace=False)
        features_training = all_features_with_EPS[index_training]

        np.save('/home/lls/stored_files/with_EPS_label/index_training.npy', index_training)
        np.save('/home/lls/stored_files/with_EPS_label/features_training.npy', features_training)

        # train algorithm

        algo = ml.MLAlgorithm(features_training)

        # Predict classes of particles in box - particles used for training

        features_left = all_features_with_EPS[~np.in1d(np.arange(len(all_features_with_EPS)), index_training)]
        predicted_probabilities = algo.classifier.predict_proba(features_left[:, :-1])
        true_labels = features_left[:,-1]

        np.save('/home/lls/stored_files/with_EPS_label/predicted_probabilities.npy',
                predicted_probabilities)
        np.save('/home/lls/stored_files/with_EPS_label/true_labels.npy',
                true_labels)

        # Get false positives and false negatives of the machine learning algorithm

        ic = parameters.InitialConditionsParameters()
        ids_all = np.concatenate((ic.ids_IN, ic.ids_OUT))
        ids = ids_all[~np.in1d(np.arange(len(ids_all)), index_training)]

        FPs, FNs = classification_results.get_FP_and_FN_ML(ids, predicted_probabilities, true_labels)

        ############# EPS PREDICTIONS ################

        # Do it tomorrow
