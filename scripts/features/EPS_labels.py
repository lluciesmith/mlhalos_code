import numpy as np
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import parameters
from mlhalos import density
from scripts.EPS import EPS_predictions

ic = parameters.InitialConditionsParameters()
ids_in = ic.ids_IN
ids_out = ic.ids_OUT

density_contrasts = density.DensityContrasts(initial_parameters=ic, num_filtering_scales=50)
delta_in = density_contrasts.density_contrast_in
delta_out = density_contrasts.density_contrast_out

EPS_label_in = EPS_predictions.EPS_label(delta_in)
EPS_label_out = EPS_predictions.EPS_label(delta_out)

np.save("/Users/lls/Documents/CODE/stored_files/all_out/EPS_label_all_in.npy", EPS_label_in)
np.save("/Users/lls/Documents/CODE/stored_files/all_out/EPS_label_all_out.npy", EPS_label_out)
