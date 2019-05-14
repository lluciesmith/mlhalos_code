"""
This is the script where I calculate the spherical overdensities features for the
initial conditions and for the z=8.89 snapshot with the periodicity implementation
of pynbody version 0.45 to calculate the smoothing including periodicity of the box.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import pynbody
import numpy as np
from mlhalos import parameters
from mlhalos import density
#
# path = "/home/lls/stored_files"
# # path = "/Users/lls/Documents/mlhalos_files"
#
# ic = parameters.InitialConditionsParameters(path=path, load_final=True)
# z_10 = pynbody.load(path + "/Nina-Simulations/double/snapshot_004")
# z_10.physical_units()
#
# d = density.DensityContrasts(initial_parameters=ic,
#                              # snapshot=z_10,
#                              num_filtering_scales=50, window_function="top hat", volume="sphere", path="/home/lls")
#
# dencon_10 = d.density_contrasts
# np.save("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy", dencon_10)
# # np.save("/share/data2/lls/features_w_periodicity_fix/z8_density_contrasts.npy", dencon_10)


# path = "/home/lls/stored_files"
# # path = "/Users/lls/Documents/mlhalos_files"
#
# ic = parameters.InitialConditionsParameters(path=path, load_final=True)
# z_01 = pynbody.load(path + "/Nina-Simulations/double/snapshot_099")
# z_01.physical_units()
#
# d = density.DensityContrasts(initial_parameters=ic,
#                              snapshot=z_01,
#                              num_filtering_scales=50, window_function="top hat", volume="sphere", path="/home/lls")
#
# dencon_01 = d.density_contrasts
# np.save("/share/data2/lls/features_w_periodicity_fix/z01_density_contrasts.npy", dencon_01)


path = "/home/lls/stored_files"
# path = "/Users/lls/Documents/mlhalos_files"

ic = parameters.InitialConditionsParameters(path=path, load_final=True)
z_0 = pynbody.load(path + "/Nina-Simulations/double/snapshot_104")
z_0.physical_units()

d = density.DensityContrasts(initial_parameters=ic,
                             snapshot=z_0,
                             num_filtering_scales=50, window_function="top hat", volume="sphere", path="/home/lls")

dencon_01 = d.density_contrasts
np.save("/share/data2/lls/features_w_periodicity_fix/z0_density_contrasts.npy", dencon_01)