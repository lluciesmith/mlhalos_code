"""
Save features of ALL particles in the box, ordered such that it is a concatenation of all in particles' features
and all out particles' features. Here, in particles live in halos between 0 to 400 and out particles live in halos
400+ or are in no halo at all.

"""

import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
import numpy as np
from mlhalos import parameters
from mlhalos import features


############### FEATURES ###############


initial_parameters = parameters.InitialConditionsParameters(min_mass_scale=1e10, max_mass_scale=1e15,
                                                            ids_type='all', num_particles=None,
                                                            n_particles_per_cat=None)
features_full_mass_scale = features.extract_labeled_features(features_type= "EPS trajectories",
                                                             initial_parameters=initial_parameters,
                                                             num_filtering_scales=50,
                                                             rescale="standard")

np.save('/Users/lls/Documents/CODE/stored_files/all_out/features_full_mass_scale.npy',
        features_full_mass_scale)
