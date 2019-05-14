"""
Predict the empirical halo masses for PS and ST for the sample variance cut trajectories.
Calculate the RADIUS first upcrossing so that you are independent of the volume definition
in this step of the analysis.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import parameters
import numpy as np
from scripts.hmf import mass_predictions_ST as mp
import pynbody
from scripts.trajectories.cut_sample_variance import sample_variance as sv

ic = parameters.InitialConditionsParameters()

d = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_above_sample_variance_min_k.npy",
            mmap_mode='r')


m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
assert r.units == 'Mpc a h**-1'
r.sim = ic.initial_conditions
r_physical = r.in_units("Mpc h**-1")

boxsize = pynbody.array.SimArray([50])
boxsize.units = "Mpc h**-1 a"
boxsize.sim = ic.initial_conditions
boxsize_physical = boxsize.in_units("Mpc h**-1")
k_min = sv.k_scale_sample_variance_power_spectrum(boxsize_physical, accuracy=0.1)

k_smoothing = 1/r_physical
index_above_sample_var = np.where(k_smoothing >= k_min)[0]

r_reliable = r_physical[index_above_sample_var]

# Can use this function with smoothing radius scale rather than smoothing masses.
# This will give the first upcrossing in terms of radius scale rather than mass.

pred_mass_all = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", cosmology="WMAP5",
                                               trajectories=den_con)

np.save("/home/lls/stored_files/trajectories_sharp_k/ALL_predicted_masses_1500_even_log_m_spaced.npy",
        pred_mass_all)