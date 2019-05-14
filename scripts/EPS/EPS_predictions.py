import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
import numpy as np
from mlhalos import parameters
from mlhalos import window
from scripts.ellipsoidal import ellipsoidal_barrier as eb

# get the trajectories in mass scale range 1e10 Msol to 1e15 Msol from NON-RESCALED features

def load_features_notrescale(features=None):
    if features is not None:
        features_test_nonrescaled = np.load(features)
    else:
        all_features_nonrescaled = np.load('/Users/lls/Documents/CODE/stored_files/all_out/'
                                            'not_rescaled/features_all_particles.npy')
        index_training = np.load('/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy')
    #     all_features_nonrescaled = np.load('/Users/lls/Documents/CODE/stored_files/not_rescaled/'
    #                                        'features_all_particles.npy')
    #     index_training = np.load('/Users/lls/Documents/CODE/stored_files/all_out/classification/50k_features_index.npy')
        features_test_nonrescaled = all_features_nonrescaled[~np.in1d(np.arange(len(all_features_nonrescaled)),
                                                                      index_training)]
    return features_test_nonrescaled


def get_in_range_indices_of_trajectories(mass_range="in", initial_parameters=None):
    if initial_parameters is None:
        initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")

    w = window.WindowParameters(initial_parameters=initial_parameters, num_filtering_scales=50)

    if mass_range == "in":
        range_mass_IN = np.intersect1d(np.where(w.smoothing_masses >
                                                initial_parameters.halo[initial_parameters.max_halo_number]['mass'].sum()),
                                       np.where(w.smoothing_masses <
                                                initial_parameters.halo[initial_parameters.min_halo_number]['mass'].sum()))
        return range_mass_IN
    else:
        NameError("Enter a valid mass range of trajectories")


def EPS_label(trajectory, mass_range="in", initial_parameters=None):
    mass_range_indices = get_in_range_indices_of_trajectories(mass_range=mass_range,
                                                              initial_parameters=initial_parameters)
    delta_sc = eb.get_spherical_collapse_barrier(initial_parameters, z=99, delta_sc_0=1.686, output="rho/rho_bar",
                                                 growth=None)
    EPS_label_trajectory = np.array([1 if any(num >= delta_sc for num in trajectory[i, mass_range_indices]) else -1
                                     for i in range(len(trajectory))])
    return EPS_label_trajectory


def get_subset_classified_particles_EPS(particles="false negatives", ids_subset=None):
    ids = np.load("/Users/lls/Documents/CODE/stored_files/all_out/classified_ids.npy")
    features_nonrescaled = load_features_notrescale()

    if ids_subset is not None:
        features_nonrescaled = features_nonrescaled[np.in1d(ids, ids_subset)]
        ids = ids_subset

    if particles == "false negatives":
        true_label = 1
        EPS_predicted_label = -1

    elif particles == "false positives":
        true_label = -1
        EPS_predicted_label = 1

    elif particles == "true positives":
        true_label = 1
        EPS_predicted_label = 1

    elif particles == "true negatives":
        true_label = -1
        EPS_predicted_label = -1

    else:
        raise NameError("Select a valid class of particles")

    trajectories_all = features_nonrescaled[features_nonrescaled[:, -1] == true_label]
    ids_all = ids[features_nonrescaled[:, -1] == true_label]

    EPS_label_trajectories_all = EPS_label(trajectories_all)
    misclassified_particles = ids_all[np.where(EPS_label_trajectories_all == EPS_predicted_label)[0]]
    return misclassified_particles


if __name__ == "__main__":

    all_features_nonrescaled = load_features_notrescale()

    trajectories_in = all_features_nonrescaled[all_features_nonrescaled[:, -1] == 1]
    trajectories_out = all_features_nonrescaled[all_features_nonrescaled[:, -1] == -1]

    EPS_label_trajectories_in = EPS_label(trajectories_in)
    EPS_label_trajectories_out = EPS_label(trajectories_out)

    num_FNs = len(np.where(EPS_label_trajectories_in == -1)[0])
    num_FPs = len(np.where(EPS_label_trajectories_out == 1)[0])

    FN_rate = num_FNs / len(trajectories_in) * 100
    FP_rate = num_FPs / len(trajectories_out) * 100

    print("The fraction of false negatives predicted by EPS is " + str(FN_rate) + "%. "
          "The fraction of false positives predicted by EPS is " + str(FP_rate) + "%.")

