"""
This is the script where I calculate the shear features for the initial conditions
with the periodicity implementation of pynbody version 0.45 to calculate
the density smoothing including periodicity of the box.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import shear
from mlhalos import parameters
import numpy as np

saving_path = "/share/data2/lls/features_w_periodicity_fix/"

ic = parameters.InitialConditionsParameters(load_final=True, path="/home/lls/stored_files")

s = shear.ShearProperties(initial_parameters=ic, num_filtering_scales=50, snapshot=None, shear_scale=range(50),
                          number_of_processors=60, path="/home/lls")

subtracted_eigenvalues = s.density_subtracted_eigenvalues
np.save(saving_path + "density_subtracted_eigenvalues.npy", subtracted_eigenvalues)

ell = s.density_subtracted_ellipticity
np.save(saving_path + "density_subtracted_ellipticity.npy", ell)

prol = s.density_subtracted_prolateness
np.save(saving_path + "density_subtracted_prolateness.npy", prol)
