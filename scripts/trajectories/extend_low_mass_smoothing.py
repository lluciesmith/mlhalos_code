"""
Save sharp-k trajectories above k = 2pi/L.

Save the "cut" trajectories as in
`` /share/data1/lls/trajectories_sharp_k/ALL_traj_above_k_min_box.npy``

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import pynbody
import numpy as np
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
from mlhalos import density
from scripts.hmf import mass_predictions_ST as mp


# Original smoothing scales

ic = parameters.InitialConditionsParameters()

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
k_smoothing = 1/r

boxsize = pynbody.array.SimArray([50])
k_min = 2*np.pi/boxsize

k_smoothing = 1/r
index_above_k_min = np.where(k_smoothing >= k_min)[0]

k_smoothing_original = k_smoothing[index_above_k_min]

# Get lowest smoothing mass scale from original smoothing scales
# and extend it down to the mass of a halo of at least 100 particles.

rho_M = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol h^2 Mpc^-3 a^-3")
m_min_original = rho_M * 6 * np.pi**2 / k_smoothing_original.max()**3

m_halo_100 = ic.initial_conditions['mass'].in_units("Msol h**-1")[0] * 100

# m_extended_smoothing = np.logspace(np.log10(m_halo_100), np.log10(m_min_original), 50)
m_extended_smoothing = np.logspace(np.log10(m_halo_100), np.log10(m_min_original), 100)

k_extended = pynbody.array.SimArray((rho_M * 6 * np.pi**2 / m_extended_smoothing)**(1/3))
k_extended.sim = ic.initial_conditions
k_extended.units = "Mpc**-1 h a**-1"

k_extended_physical = k_extended.in_units("Mpc**-1")

# Smooth the density field with a sharp-k window function
# at smoothign scales `k_extended_physical`

r_extended_smoothing = 1/k_extended_physical

den = density.Density(initial_parameters=ic, window_function="sharp k")
smoothed_density = den.get_smooth_density_for_radii_list(ic, r_extended_smoothing)

den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density)

# np.save("/share/data1/lls/trajectories_sharp_k/extended_traj_low_mass.npy", den_con)
np.save("/share/data1/lls/trajectories_sharp_k/extended_traj_low_mass_100_scales.npy", den_con)



