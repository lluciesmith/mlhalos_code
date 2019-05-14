"""
In this script, I train the algorithm with a training set such that:
- The features are EPS trajectories evaluated at 50 log-spaced mass scales of range 1.8362e+12 Msol < M < 4.1489e+14
    Msol.
- The samples are are 50,000 random particles taken from the full set of IN particles ( particles in halos 0 to 400)
    and OUT particles ( particles in halos 400+ and particles in no halos.)

The algorithm is used to predict probabilities of classes for all remaining particles in the box.
The feature importances of the training set are also store in the `all_out` folder.

"""

import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from mlhalos import machinelearning as ml
from mlhalos import features
from mlhalos import parameters
from mlhalos import plot
import numpy as np


# Select 50,000 particles to use as training set

ic_all = parameters.InitialConditionsParameters()

min_halo = 0
max_halo = 400
min_mass = ic_all.halo[max_halo]['mass'].sum()
max_mass = ic_all.halo[min_halo]['mass'].sum()
ic = parameters.InitialConditionsParameters(min_halo_number=min_halo, max_halo_number=max_halo, min_mass_scale=min_mass,
                                            max_mass_scale=max_mass)

feat_w_EPS = features.extract_labeled_features(initial_parameters=ic,
                                               add_EPS_label=True, n_samples=50000)




# Train the algorithm

algorithm_50k = ml.MLAlgorithm(features_training)

np.save('/Users/lls/Documents/CODE/stored_files/all_out/train_set_x.npy',
        algorithm_50k.X_train)
np.save('/Users/lls/Documents/CODE/stored_files/all_out/test_set_x.npy',
        algorithm_50k.X_test)
np.save('/Users/lls/Documents/CODE/stored_files/all_out/train_set_y.npy',
        algorithm_50k.y_train)
np.save('/Users/lls/Documents/CODE/stored_files/all_out/test_set_y.npy',
        algorithm_50k.y_true)


# Make predictions on all particles other than the training set

features_left = features_full_mass[~np.in1d(np.arange(len(features_full_mass)), index_training)]
predicted_probabilities = algorithm_50k.classifier.predict_proba(features_left[:, :-1])
true_labels = features_left[:,-1]

np.save('/Users/lls/Documents/CODE/stored_files/all_out/predicted_probabilities.npy',
        predicted_probabilities)
np.save('/Users/lls/Documents/CODE/stored_files/all_out/true_labels.npy',
        true_labels)


# Plot feature importance

importance = algorithm_50k.classifier.best_estimator_.feature_importances_
indices = np.argsort(importance)[::-1]

np.save('/Users/lls/Documents/CODE/stored_files/all_out/importance.npy',
        importance)
np.save('/Users/lls/Documents/CODE/stored_files/all_out/indices.npy',
        indices)

figure = plot.plot_feature_importance(importance, indices)
figure.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feat_importance.png")