import numpy as np
import pynbody
import sys
sys.path.append('/Users/lls/Documents/CODE/git/mlhalos_code/scripts')


def load_classification_results():
    # results = np.load('/Users/lls/Documents/CODE/stored_files/classification/classification_results.npy')
    results = np.load('/Users/lls/Documents/CODE/stored_files/all_out/classification_results.npy')
    ids = results[:, 0]
    true_labels = results[:, 1]
    predicted = results[:, 2:4]
    return ids, predicted, true_labels


def load_final_snapshot_and_halos():
    f = pynbody.load("/Users/lls/Documents/CODE/Nina-Simulations/double/snapshot_104")
    f.physical_units()
    h = f.halos(make_grp=True)
    return f, h


# def get_predicted_and_true_labels(ids, training_set_size=2000):
#     """ ids need to be in form (ids_in,ids_out) in same order as all_ids"""
#
#     # load all particles ids and corresponding predicted and true labels
#
#     all_in_ids = np.load('/Users/lls/Documents/CODE/stored_files/all_particles/all_in_ids.npy')
#     all_out_ids = np.load('/Users/lls/Documents/CODE/stored_files/all_particles/all_out_ids.npy')
#     all_ids = np.concatenate((all_in_ids, all_out_ids))
#
#     if all_ids.dtype != "int":
#         all_ids = all_ids.astype('int')
#
#     if training_set_size == 2000:
#         all_predicted = np.load('/Users/lls/Documents/CODE/stored_files/all_particles'
#                                 '/algo_2000_training/predicted_probabilities.npy')
#     elif training_set_size == 50000:
#         all_predicted = np.load('/Users/lls/Documents/CODE/stored_files/all_particles/algo_50000_training'
#                                 '/predicted_probabilities.npy')
#     else:
#         raise ValueError("Select a valid training set size: either 50000 or 2000.")
#
#     all_true = np.load('/Users/lls/Documents/CODE/stored_files/all_particles/true_labels.npy')
#
#     # extract predicted and true labels of ids we're interested in
#
#     y_predicted = all_predicted[np.in1d(all_ids, ids)]
#     y_true = all_true[np.in1d(all_ids, ids)]
#     return y_predicted, y_true


def get_category_particle_ids(category="false negatives", threshold=None, ids=None, y_predicted=None, y_true=None):
    if ids is None and y_predicted is None and y_true is None:
        ids, y_predicted, y_true = load_classification_results()

    if category == "false positives":
        particles = get_false_positives(ids, y_predicted, y_true, threshold)
    elif category == "false negatives":
        particles = get_false_negatives(ids, y_predicted, y_true, threshold)
    elif category == "true positives":
        particles = get_true_positives(ids, y_predicted, y_true, threshold)
    elif category == "true negatives":
        particles = get_true_negatives(ids, y_predicted, y_true, threshold)

    return particles


def get_false_positives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] > threshold
    y_bool = (y_true == 1)

    FP = ids[labels & ~y_bool]
    return FP


def get_false_negatives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] > threshold
    y_bool = (y_true == 1)

    FN = ids[~labels & y_bool]
    return FN


def get_true_negatives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] > threshold
    y_bool = (y_true == 1)

    TN = ids[~labels & ~y_bool]
    return TN


def get_true_positives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] > threshold
    y_bool = (y_true == 1)

    TP = ids[labels & y_bool]
    return TP


# def get_false_positive_false_negative_rates(y_predicted, y_true):
#
#     labels = []
#     for i in range(len(y_predicted)):
#         if y_predicted[i][0] > y_predicted[i][1]:
#             labels.append(False)
#         elif y_predicted[i][0] < y_predicted[i][1]:
#             labels.append(True)
#
#     labels = np.array(labels)
#
#     y_bool = (y_true == 1)
#
#     FP = (labels & ~y_bool).sum(axis=0)
#     FN = (~labels & y_bool).sum(axis=0)
#
#     return FP, FN


################ HALOS PROPERTIES FUNCTIONS ################

def get_pure_ids_and_halos(initial_conditions, ids, y_pred, probability_threshold):

    proba_in = y_pred[:, 1]
    high_proba_ids = ids[proba_in > probability_threshold]

    halos_hp = initial_conditions.final_snapshot['grp'][high_proba_ids]
    halos_hp_mass = [initial_conditions.halo[i]['mass'].sum() for i in halos_hp]

    return high_proba_ids, halos_hp_mass



def get_ratio_ids_all_per_bin(n_ids, n_all):
    # calculate ratio for n_all!=0

    ratio = n_ids[n_all!=0]/n_all[n_all!=0]
    return ratio


def get_halo_mass(ids, f=None, log=True):
    if f is None:
        f = pynbody.load("/Users/lls/Documents/CODE/Nina-Simulations/double/snapshot_104")
    f.physical_units()
    h = f.halos(make_grp=True)

    if ids.dtype != 'int':
        ids = ids.astype('int')

    halos_ID = f[ids]['grp']
    if log is True:
        halos_mass = np.array([np.log10(h[i]['mass'].sum()) for i in halos_ID])
    else:
        halos_mass = np.array([h[i]['mass'].sum() for i in halos_ID])

    return halos_mass


def split_halos_mass_in_bins(halos_mass, bins=30, normed=False):
    n, b = np.histogram(halos_mass, bins=bins, normed=normed)
    return n, b





