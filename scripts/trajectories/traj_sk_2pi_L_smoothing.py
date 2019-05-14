"""
Smooth the density field with a sharp-k filter at characteristic scales in range
2pi/L and (2pi/L)(N/2) in steps of 2pi/L.

Save the trajectories in
`` /share/data1/lls/trajectories_sharp_k/traj_smoothed_multiple_2piL.npy``

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import parameters
import numpy as np
from mlhalos import density
import pynbody


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


ic = parameters.InitialConditionsParameters()

# L is in physical units Mpc^-1 at z=99
L = ic.boxsize_no_units
delta_k = 2 * np.pi / L
nyquist = delta_k * ic.shape/2
spacing = np.sqrt(3)/3 * delta_k

k_smoothing = np.arange(delta_k, nyquist + spacing, spacing)
r_smoothing = 1/k_smoothing

den = density.Density(initial_parameters=ic, window_function="sharp k")
smoothed_density = den.get_smooth_density_for_radii_list(ic, r_smoothing)

den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
np.save("/home/lls/stored_files/trajectories_sharp_k/traj_smoothed_multiple_2piL.npy", den_con)


# Actually extract from trajectories you already have

from scripts.hmf import hmf_tests as ht

m_bins = 10 ** np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"

r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
assert r.units == 'Mpc a h**-1'
r.sim = ic.initial_conditions
r_physical = r.in_units("Mpc h**-1")

k_smoothing = 1 / r_physical
delta_k = 2*np.pi/ic.boxsize_no_units
ksm_orig = k_smoothing/delta_k

spacing = np.sqrt(3)/3 * delta_k
k_scales = np.arange(delta_k, nyquist + spacing, spacing)


k_near_multiples = np.array([find_nearest(ksm_orig, value) for value in k_scales]).view(pynbody.array.SimArray)
