import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import inertia

saving_path = "/share/data1/lls/reseed50/features/inertia/"
path_simulation = "/share/data1/lls/reseed50/"

initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_simulation + "IC.gadget3",
                                                        load_final=False, min_halo_number=0, max_halo_number=400,
                                                        min_mass_scale=3e10, max_mass_scale=1e15)
d = np.load(path_simulation + "features/density_contrasts.npy")
delta = d.transpose()
assert delta.shape == (50, 16777216)

In = inertia.Inertia(initial_parameters=initial_params, density_contrasts=delta, num_filtering_scales=50)
eig_0, eig_1, eig_2 = In.get_eigenvalues_inertia_tensor_from_multiple_densities(delta, number_of_processors=40)

np.save(saving_path + "eigenvalues_0.npy", eig_0)
np.save(saving_path + "eigenvalues_1.npy", eig_1)
np.save(saving_path + "eigenvalues_2.npy", eig_2)

