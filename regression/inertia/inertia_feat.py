import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import inertia

# saving_path = "/share/data1/lls/regression/inertia/cores_40/"
# path_trajectories = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
#
# ic = parameters.InitialConditionsParameters(load_final=True)
# d = np.load(path_trajectories + "density_trajectories.npy")
# delta = d.transpose()
# assert delta.shape == (50, 16777216)
#
# In = inertia.Inertia(initial_parameters=ic, density_contrasts=delta, num_filtering_scales=50)
# eig_0, eig_1, eig_2 = In.get_eigenvalues_inertia_tensor_from_multiple_densities(delta, number_of_processors=40)
#
# np.save(saving_path + "eigenvalues_0.npy", eig_0)
# np.save(saving_path + "eigenvalues_1.npy", eig_1)
# np.save(saving_path + "eigenvalues_2.npy", eig_2)

"""
RE-RUN the feature extraction with density calculated with periodicity fix of
pynbody version 0.45

"""

path_features = "/share/data2/lls/features_w_periodicity_fix/"
saving_path = "/share/data2/lls/features_w_periodicity_fix/inertia/"

ic = parameters.InitialConditionsParameters(load_final=True)
d = np.load(path_features + "ics_density_contrasts.npy")
delta = d.transpose()
assert delta.shape == (50, 16777216)

In = inertia.Inertia(initial_parameters=ic, density_contrasts=delta, num_filtering_scales=50)
eig_0, eig_1, eig_2 = In.get_eigenvalues_inertia_tensor_from_multiple_densities(delta, number_of_processors=40)

np.save(saving_path + "eigenvalues_0.npy", eig_0)
np.save(saving_path + "eigenvalues_1.npy", eig_1)
np.save(saving_path + "eigenvalues_2.npy", eig_2)
