"""
Save features of ALL particles in the box, ordered such that it is a concatenation of all in particles' features
and all out particles' features. Here, in particles live in halos between 0 to 400 and out particles live in halos
400+ or are in no halo at all.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import features


############### FEATURES ###############

path = "/home/lls/stored_files"
number_of_cores = 60

initial_parameters = parameters.InitialConditionsParameters(min_mass_scale=3e10, max_mass_scale=1e15,
                                                            ids_type='all', num_particles=None,
                                                            n_particles_per_cat=None)
features_all_particles = features.extract_labeled_features(features_type=["EPS trajectories", "shear"],
                                                           initial_parameters=initial_parameters, num_filtering_scales=50,
                                                           rescale=None, shear_scale=range(50),
                                                           density_subtracted_shear=False, path=path,
                                                           cores=number_of_cores)

np.save('/home/lls/stored_files/shear_no_rescaling/features.npy', features_all_particles)

density_ellipticity_features = np.column_stack((features_all_particles[:, :100], features_all_particles[:, -1]))

index_training = np.load("/home/lls/stored_files/50k_features_index.npy")
training_density_ellipticity = density_ellipticity_features[index_training, :]

index_test = np.random.choice(len(features_all_particles), 5000)
index_test = index_test[~np.in1d(index_test, index_training)]

testing_density_ellipticity_5000 = density_ellipticity_features[index_test, :]

np.save('/home/lls/stored_files/shear_no_rescaling/density_ellipticity_features.npy', density_ellipticity_features)
np.save('/home/lls/stored_files/shear_no_rescaling/training_density_ellipticity_features.npy',
        training_density_ellipticity)
np.save('/home/lls/stored_files/shear_no_rescaling/testing_density_ellipticity_features.npy',
        testing_density_ellipticity_5000)
