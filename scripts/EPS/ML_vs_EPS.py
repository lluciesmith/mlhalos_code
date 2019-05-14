"""
RUN ON HYPATIA

(1)I compute the NON_RESCALED features for all particles in the simulation box and save them in
`all_non_rescaled_features.npy`

(2) I train the algorithm with a training set such that:
- The features are EPS trajectories evaluated at 50 log-spaced mass scales of range 1e10 Msol < M < 1e15 Msol.
- The training samples are 50,000 random particles taken from the full set of IN particles ( particles in halos 0 to 400)
    and OUT particles ( particles in halos 400+ and particles in no halos.)

(3) I test the algorithm's performance on the remainning particles in the box.
(4) I compare the algorithm's predictions to the predictions from EPS theory.

"""


import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
import numpy as np
from mlhalos import parameters
from mlhalos import features
from mlhalos import machinelearning as ml


#### FEATURE EXTRACTION #####

initial_parameters = parameters.InitialConditionsParameters(min_mass_scale=1e10, max_mass_scale=1e15,
                                                            ids_type='all', num_particles=None,
                                                            n_particles_per_cat=None)
features_all = features.extract_labeled_features(features_type="EPS trajectories",
                                                 initial_parameters=initial_parameters,
                                                 num_filtering_scales=50,
                                                 rescale=None)

np.save('/home/lls/stored_files/non_rescaled/features_all_particles.npy',
        features_all)


#### TRAINING #####
# Select 50,000 particles to use as training set

index_training = np.random.choice(len(features_all), size=50000, replace=False)
features_training = features_all[index_training]

np.save("/home/lls/stored_files/non_rescaled/features_training_index.npy", index_training)
np.save("/home/lls/stored_files/non_rescaled/50k_features_training.npy", features_training)


# Train the algorithm

RF = ml.MLAlgorithm(features_training, split_data_method=None, n_jobs=24)


#### PREDICT PROBABILITIES ON REMAINING PARTICLES IN THE BOX ######

features_left = features_all[~np.in1d(np.arange(len(features_all)), index_training)]

predicted_probabilities = RF.classifier.predict_proba(features_left[:, :-1])
true_labels = features_left[:,-1]

np.save('/home/lls/stored_files/non_rescaled/pred_proba_features_left.npy',
        predicted_probabilities)
np.save('/home/lls/stored_files/non_rescaled/rue_labels_features_left.npy',
        true_labels)


#### EPS PREDICTIONS ON REMAINING PARTICLES IN THE BOX ######

trajectories_in = features_left[features_left[:, -1] == 1]
trajectories_out = features_left[features_left[:, -1] == -1]

EPS_label_trajectories_in = []
EPS_label_trajectories_out = []

for i in range(len(trajectories_in)):
    if any(num >= 1.0169 for num in trajectories_in[i]):
        EPS_label_trajectories_in.append(1)
    else:
        EPS_label_trajectories_in.append(-1)

for i in range(len(trajectories_out)):
    if any(num >= 1.0169 for num in trajectories_out[i]):
        EPS_label_trajectories_out.append(1)
    else:
        EPS_label_trajectories_out.append(-1)

np.save('/home/lls/stored_files/non_rescaled/EPS_predictions_in.npy', np.array(EPS_label_trajectories_in))
np.save('/home/lls/stored_files/non_rescaled/EPS_predictions_out.npy',np.array(EPS_label_trajectories_out))
