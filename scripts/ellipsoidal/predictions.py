import numpy as np
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
from mlhalos import window
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb


def get_in_range_indices_of_trajectories(mass_range="in", ic=None, w=None):
    if ic is None:
        ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")

    if w is None:
        w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)

    if mass_range == "in":
        range_mass_IN = np.intersect1d(np.where(w.smoothing_masses >= ic.halo[ic.max_halo_number]['mass'].sum()),
                                       np.where(w.smoothing_masses <= ic.halo[ic.min_halo_number]['mass'].sum()))
        return range_mass_IN
    else:
        NameError("Enter a valid mass range of trajectories")


def get_predicted_ellipsoidal_label(trajectory, threshold_barrier, mass_range_indices):
    IN_threshold_barrier = threshold_barrier[mass_range_indices]
    densities_IN_mass_range = trajectory[:, mass_range_indices]

    ellipsoidal_label = np.array([1 if (densities_IN_mass_range[i] >= IN_threshold_barrier).any()
                                  else -1 for i in range(len(trajectory))])
    return ellipsoidal_label


def ellipsoidal_collapse_predicted_label(trajectory, mass_range="in", initial_parameters=None,
                                         window_parameters=None, beta=0.485, gamma=0.615, a=0.707):
    if initial_parameters is None:
        initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")
    if window_parameters is None:
        window_parameters = window.WindowParameters(initial_parameters=initial_parameters, num_filtering_scales=50)

    mass_range_indices = get_in_range_indices_of_trajectories(mass_range=mass_range, ic=initial_parameters,
                                                              w=window_parameters)

    ellip_threshold = eb.ellipsoidal_collapse_barrier(window_parameters.smoothing_masses, initial_parameters,
                                                      beta=beta, gamma=gamma, a=a, z=99)
    ellipsoidal_label = get_predicted_ellipsoidal_label(trajectory, ellip_threshold, mass_range_indices)

    return ellipsoidal_label


def get_fpr_tpr_ellipsoidal_prediction(ellipsoidal_predicted_label, true_label):
    class_label = (true_label == 1)
    predicted_label = (ellipsoidal_predicted_label == 1)

    fpr = (predicted_label & ~class_label).sum() / len(np.where(true_label == -1)[0])
    tpr = (predicted_label & class_label).sum() / class_label.sum()

    return fpr, tpr


def get_fpr_tpr_indices(predicted_label, true_label):
    class_label = (true_label == 1)
    predicted_label = (predicted_label == 1)

    fpr_indices = (predicted_label & ~class_label)
    tpr_indices = (predicted_label & class_label)

    return fpr_indices, tpr_indices


def get_fpr_tpr_from_features(features, mass_range="in", initial_parameters=None,window_parameters=None, beta=0.485,
                              gamma=0.615, a=0.707):
    predicted_labels = ellipsoidal_collapse_predicted_label(features[:, :-1], mass_range=mass_range,
                                                            initial_parameters=initial_parameters,
                                                            window_parameters=window_parameters, beta=beta,
                                                            gamma=gamma, a=a)
    true_labels = features[:, -1]
    fpr, tpr = get_fpr_tpr_ellipsoidal_prediction(predicted_labels, true_labels)
    return fpr, tpr


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)
    den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")

    training_index = np.load('/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy')
    den_f_test = den_f[~np.in1d(np.arange(len(den_f)), training_index)]

    # ell_label_075 = ellipsoidal_collapse_predicted_label(den_f[:, :-1], initial_parameters=ic, window_parameters=w,
    #                                                  beta=0.5, gamma=0.6, a=0.75)
    ell_label_0707 = ellipsoidal_collapse_predicted_label(den_f_test[:, :-1], initial_parameters=ic, window_parameters=w,
                                                          beta=0.485, gamma=0.615, a=0.707)
    # ell_label_1 = ellipsoidal_collapse_predicted_label(den_f[:, :-1], initial_parameters=ic, window_parameters=w,
    #                                                  beta=0.485, gamma=0.615, a=1)
    true_label = den_f_test[:, -1]

    fpr_0707, tpr_0707 = get_fpr_tpr_ellipsoidal_prediction(ell_label_0707, true_label)
    #fpr_0707, tpr_0707 = get_fpr_tpr_ellipsoidal_prediction(ell_label_0707, true_label)
    print("FPR is " + str(fpr_0707) + " and TPR is " + str(tpr_0707))