import numpy as np

from mlhalos import trajectories
from utils import classification_results


############## COMPARISON EPS AND MLALGO PREDICTIONS FUNCTIONS ################


def get_trajectories_of_ids(ids=None, ids_in=None, ids_out=None):
    all_density_contrast_in = np.load(
        '/Users/lls/Documents/CODE/stored_files/all_particles/all_in_density_contrasts.npy')
    all_density_contrast_out = np.load(
        '/Users/lls/Documents/CODE/stored_files/all_particles/all_out_density_contrasts.npy')

    all_in_ids = np.load('/Users/lls/Documents/CODE/stored_files/all_particles/all_in_ids.npy')
    all_out_ids = np.load('/Users/lls/Documents/CODE/stored_files/all_particles/all_out_ids.npy')

    if ids_in is not None and ids_out is not None:
        density_contrast_in = all_density_contrast_in[np.in1d(all_in_ids, ids_in)]
        density_contrast_out = all_density_contrast_out[np.in1d(all_out_ids, ids_out)]
    else:
        density_contrast_in = all_density_contrast_in[np.in1d(all_in_ids, ids)]
        density_contrast_out = all_density_contrast_out[np.in1d(all_out_ids, ids)]

    return density_contrast_in, density_contrast_out


def compare_EPS_with_algo_predictions(ids):
    # get TRAJECTORIES of ids

    density_contrast_in, density_contrast_out = get_trajectories_of_ids(ids)

    in_traj_no_crossing = trajectories.get_num_particles_that_never_cross_threshold(density_contrast_in)
    out_traj_crossing = trajectories.get_num_particles_that_cross_the_threshold(density_contrast_out)

    print("The number of in particles not crossing the trajectories is " + str(in_traj_no_crossing)
          + " out of " + str(len(density_contrast_in)) + " and the number of out particles crossing the threshold is "
          + str(out_traj_crossing) + " out of " + str(len(density_contrast_out)))

    # FALSE POSITIVES and FALSE NEGATIVES
    y_predicted, y_true = classification_results.get_predicted_and_true_labels(ids)

    labels = []
    for i in range(len(y_predicted)):
        if y_predicted[i][0] > y_predicted[i][1]:
            labels.append(False)
        elif y_predicted[i][0] < y_predicted[i][1]:
            labels.append(True)

    labels = np.array(labels)

    y_bool = (y_true == 1)

    TP = ids[labels & y_bool]
    FP = ids[labels & ~y_bool]
    FN = ids[~labels & y_bool]
    TN = ids[~labels & ~y_bool]

    print("The number of false negatives is " + str(len(FN)) + " out of " + str(len(FN) + len(TP))
          + " positives and the number of false positives is " + str(len(FP)) + " out of " + str(len(FP) + len(TN))
          + " negatives.")


def get_FP_and_FN_EPS(ids=None, ids_in=None, ids_out=None, density_contrast_in=None, density_contrast_out=None):

    if density_contrast_in is None and density_contrast_out is None:
        density_contrast_in, density_contrast_out = get_trajectories_of_ids(ids=ids, ids_in=ids_in, ids_out=ids_out)

    FP_EPS_index = trajectories.get_index_particles_that_cross_the_threshold(density_contrast_out)
    FN_EPS_index = trajectories.get_index_particles_that_never_cross_threshold(density_contrast_in)

    FP_EPS = ids[FP_EPS_index]
    FN_EPS = ids[FN_EPS_index]

    return FP_EPS, FN_EPS


def get_pred(ids, y_predicted, y_true):
    if len(y_predicted.shape) == 1:
        labels = (y_predicted == 1)
    else:
        labels = y_predicted[:,0] < y_predicted[:,1]
    y_bool = (y_true == 1)

    TP = ids[labels & y_bool]
    FP = ids[labels & ~y_bool]
    FN = ids[~labels & y_bool]
    TN = ids[~labels & ~y_bool]

    print("The number of false negatives is " + str(len(FN)) + " out of " + str(len(FN) + len(TP))
          + " positives and the number of false positives is " + str(len(FP)) + " out of " + str(len(FP) + len(TN))
          + " negatives.")
    return TP, FP, FN, TN