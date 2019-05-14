import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import window
from mlhalos import parameters
import pynbody
from scripts.hmf import hmf

# ic = parameters.InitialConditionsParameters()
# w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250)
#
# d = density.Density(initial_parameters=ic, num_filtering_scales=250)
# densities = d._density
#
# np.save("/home/lls/stored_files/densities_250_bins.npy", densities)

ic = parameters.InitialConditionsParameters()
w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250)

d = np.load("/home/lls/stored_files/densities_250_bins.npy")
rho_m = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol Mpc**-3")

den_con = d/rho_m
del d

log_mass, mid_mass = hmf.get_log_bin_masses_and_mid_values(initial_parameters=ic, window_parameters=w)
delta_log_mass = log_mass[2] - log_mass[1]

n_density_halos_PS = hmf.get_predicted_number_density_halos_in_mass_bins(log_mass, mid_mass, kernel="PS",
                                                                         initial_parameters=ic, trajectories=den_con)
PS_dn_dlog10m = hmf.get_dn_dlog10m(n_density_halos_PS, 10 ** log_mass, 10 ** mid_mass)

np.save("/home/lls/stored_files/PS_dn_dlog10m_250_bins.npy", PS_dn_dlog10m)
