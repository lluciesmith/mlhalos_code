import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import shear
from mlhalos import parameters
import numpy as np

path = "/home/lls/stored_files"

ic = parameters.InitialConditionsParameters(initial_snapshot=
                                            path+"/Nina-Simulations/double/ICs_z99_256_L50_gadget3.dat",
                                            final_snapshot=path+"/Nina-Simulations/double/snapshot_104")

scales = range(50)
s = shear.Shear(initial_parameters=ic, num_filtering_scales=50,
                snapshot=None, shear_scale=scales, density_particles=None, number_of_processors=60, path=path)

eigenvalues = s.shear_eigenvalues
subtracted_eigenvalues = s.density_subtracted_eigenvalues

np.save(path + "/shear_eigenvalues/all_eigenvalues.npy", eigenvalues)
np.save(path + "/shear_eigenvalues/all_density_subtracted_eigenvalues.npy", subtracted_eigenvalues)
