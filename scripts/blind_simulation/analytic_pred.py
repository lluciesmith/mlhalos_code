"""
Blind simulation:
ICs - /home/app/reseed/snapshot_099
Final snapshot - /home/app/reseed/IC.gadget3

"""

import sys
sys.path.append('/home/lls/mlhalos_code')
import numpy as np
from mlhalos import parameters
from scripts.EPS import EPS_predictions as EPS_pred
from scripts.ellipsoidal import predictions as ST_pred


############### ROC CURVES ###############

if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(initial_snapshot="/home/app/reseed/IC.gadget3",
                                                final_snapshot="/home/app/reseed/snapshot_099",
                                                load_final=True, min_halo_number=0, max_halo_number=400,
                                                min_mass_scale=3e10, max_mass_scale=1e15)

    # Change ic.ids_IN and ic.ids_OUT to be IN or OUT depending on whether they are in halos of mass larger than the
    # mass of halo 400 in ic_training and not that of halo 400 in ic. Change ids_IN, ids_OUT, and ic.max_halo_number.
    # We have that halo 409 in ic has the same mass as halo 400 in ic_training - hard code this for now.

    ic.max_halo_number = 409

    density_features = np.load("/share/data1/lls/reseed50/features/density_contrasts.npy")
    EPS_predicted_label = EPS_pred.EPS_label(density_features, initial_parameters=ic)
    np.save("/share/data1/lls/reseed50/predictions/EPS_predicted_label.npy", EPS_predicted_label)

    ST_predicted_label = ST_pred.ellipsoidal_collapse_predicted_label(density_features, initial_parameters=ic,
                                                                      beta=0.485, gamma=0.615, a=0.707)
    np.save("/share/data1/lls/reseed50/predictions/ST_predicted_label.npy", ST_predicted_label)


