"""
Planck-like cosmology simulation pioneer50:
ICs - /share/data1/lls/pioneer50/pioneer50.512.ICs.tipsy
Final snapshot - /share/data1/lls/pioneer50/pioneer50.512.004096

"""

import sys
sys.path.append('/home/lls/mlhalos_code')
import numpy as np
from mlhalos import parameters
from scripts.EPS import EPS_predictions as EPS_pred
from scripts.ellipsoidal import predictions as ST_pred
from mlhalos import window
from scripts.ellipsoidal import ellipsoidal_barrier as eb

############### ROC CURVES ###############

if __name__ == "__main__":

    path_sim = "/share/data1/lls/pioneer50/"
    path = "/share/data1/lls/pioneer50/features_3e10/"
    ic = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "pioneer50.512.ICs.tipsy",
                                                final_snapshot=path_sim + "pioneer50.512.004096",
                                                load_final=True, min_halo_number=1, max_halo_number=134,
                                                min_mass_scale=3e10, max_mass_scale=1e15, sigma8=0.831)

    # Change ic.ids_IN and ic.ids_OUT to be IN or OUT depending on whether they are in halos of mass larger than the
    # mass of halo 400 in ic_training and not that of halo 400 in ic. Change ids_IN, ids_OUT, and ic.max_halo_number.
    # We have that halo 409 in ic has the same mass as halo 400 in ic_training - hard code this for now.

    # density_features = np.load("/share/data1/lls/pioneer50/features/density_contrasts.npy")
    #
    # EPS_predicted_label = EPS_pred.EPS_label(density_features, initial_parameters=ic)
    # np.save("/share/data1/lls/pioneer50/predictions/EPS_predicted_label.npy", EPS_predicted_label)
    #
    # ST_predicted_label = ST_pred.ellipsoidal_collapse_predicted_label(density_features, initial_parameters=ic,
    #                                                                   beta=0.485, gamma=0.615, a=0.707)
    # np.save("/share/data1/lls/pioneer50/predictions/ST_predicted_label.npy", ST_predicted_label)

    window_parameters = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)
    mass_range_indices = ST_pred.get_in_range_indices_of_trajectories(mass_range="in", ic=ic, w=window_parameters)
    ellip_threshold = eb.ellipsoidal_collapse_barrier(window_parameters.smoothing_masses, ic, beta=0.485,
                                                      gamma=0.615, a=0.707, z=99, cosmology="PLANCK")

    density_features = np.lib.format.open_memmap(path + "density_contrasts.npy",
                                                 mode="r", shape=(512 ** 3, 50))

    b = np.array_split(np.arange(512 ** 3), 10000, axis=0)

    for i in range(10000):
        d = density_features[b[i], :]
        EPS_i = EPS_pred.EPS_label(d, initial_parameters=ic)
        np.save(path + "predictions/EPS/EPS_pred_" + str(i) + ".npy", EPS_i)

        ST_i = ST_pred.get_predicted_ellipsoidal_label(d, ellip_threshold, mass_range_indices)
        np.save(path + "predictions/ST/ST_pred_" + str(i) + ".npy", ST_i)

    EPS_all = np.zeros((512**3,))
    ST_all = np.zeros((512**3,))

    for i in range(10000):
        EPS_ij = np.load(path + "predictions/EPS/EPS_pred_" + str(i) + ".npy")
        EPS_all[b[i]] = EPS_ij
        del EPS_ij

        ST_ij = np.load(path + "predictions/ST/ST_pred_" + str(i) + ".npy")
        ST_all[b[i]] = ST_ij
        del ST_ij

    np.save(path + "predictions/EPS_predicted_label.npy", EPS_all)
    np.save(path + "predictions/ST_predicted_label.npy", ST_all)
