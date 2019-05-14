import sys
sys.path.append("/home/lls/mlhalos_code/scripts")

import numpy as np
from mlhalos import features
from mlhalos import parameters

path = "/home/lls/stored_files"
number_of_cores = 60

# if path == "hypatia":
#     path = "/home/lls/stored_files"
# elif path == "macpro":
#     sys.path.append("/Users/lls/Documents/mlhalos_code/scripts")
#     path = "/Users/lls/Documents/CODE"

ic = parameters.InitialConditionsParameters(initial_snapshot=
                                            path+"/Nina-Simulations/double/ICs_z99_256_L50_gadget3.dat",
                                            final_snapshot=path+"/Nina-Simulations/double/snapshot_104")

density_shear_features = features.extract_labeled_features(features_type=["EPS trajectories", "shear"],
                                                           initial_parameters=ic, num_filtering_scales=50,
                                                           rescale=None, shear_scale=range(50),
                                                           density_subtracted_shear=False, path=path,
                                                           cores=number_of_cores)

np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/density_full_shear_features.npy",
        density_shear_features)
