import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
from mlhalos import density
from mlhalos import parameters

path = "/home/lls/stored_files"
number_of_cores = 60

ic = parameters.InitialConditionsParameters(initial_snapshot=
                                            path+"/Nina-Simulations/double/ICs_z99_256_L50_gadget3.dat",
                                            final_snapshot=path+"/Nina-Simulations/double/snapshot_104",
                                            load_final=True)

d = density.DensityContrasts(initial_parameters=ic, num_filtering_scales=50, path=None, window_function="top hat",
                             volume="sphere")
np.save("/home/lls/stored_files/trajectories_test.npy", d.density_contrasts)