"""
WMAP5-like cosmology simulation sim200:
ICs - /share/data1/lls/sim200/simulation/standard200.gadget3
Final snapshot - /share/data1/lls/sim200/simulation/snapshot_011

"""

import sys
sys.path.append('/home/lls/mlhalos_code')
import numpy as np
from mlhalos import parameters
from mlhalos import density
from mlhalos import shear


############### FEATURE EXTRACTION ###############

if __name__ == "__main__":
    path = "/home/lls"
    ics = "/share/data1/lls/sim200/simulation/standard200.gadget3"
    f = "/share/data1/lls/sim200/simulation/snapshot_011"
    initial_params = parameters.InitialConditionsParameters(initial_snapshot=ics, final_snapshot=f,
                                                            load_final=True, min_halo_number=0, max_halo_number=400,
                                                            #min_mass_scale=5e9, max_mass_scale=1e15
                                                            min_mass_scale=3e10, max_mass_scale=1e15
                                                            )

    sp = shear.ShearProperties(initial_parameters=initial_params, num_filtering_scales=50, shear_scale="all",
                               number_of_processors=60, path=path)

    ell = sp.density_subtracted_ellipticity
    prol = sp.density_subtracted_prolateness

    np.save("/share/data1/lls/pioneer50/features_3e10/density_subtracted_ellipticity.npy", ell)
    np.save("/share/data1/lls/pioneer50/features_3e10/density_subtracted_prolateness.npy", prol)

    Dc = density.DensityContrasts(initial_parameters=initial_params, num_filtering_scales=50, path=path)
    density_cont = Dc.density_contrasts

    np.save("/share/data1/lls/pioneer50/features_3e10/density_contrasts.npy", density_cont)
    del density_cont
    del Dc