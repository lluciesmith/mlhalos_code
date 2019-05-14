import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
from scripts.hmf import predict_masses as mp


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

# m_extended_smoothing = pynbody.array.SimArray(np.logspace(np.log10(m_halo_100), np.log10(m_min_original), 50))
m_extended_smoothing = pynbody.array.SimArray(np.logspace(np.log10(m_halo_100), np.log10(m_min_original), 100))
m_extended_smoothing.units = "Msol h**-1"

# traj = np.load("/share/data1/lls/trajectories_sharp_k/extended_traj_low_mass.npy")
traj = np.load("/share/data1/lls/trajectories_sharp_k/extended_traj_low_mass_100_scales.npy")
assert traj.shape == (256**3, len(m_extended_smoothing))


# Find trajectories which have not upcrossed for the 1500 smoothing scales case
# and predict their masses for PS and ST

# PS
pred_original_PS = np.load("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/"
                           "ALL_predicted_masses_1500_even_log_m_spaced.npy")
nan_index_PS = np.isnan(pred_original_PS)

pred_t_PS = mp.get_predicted_analytic_mass(m_extended_smoothing, ic, barrier="spherical",
                                           trajectories=traj[nan_index_PS, :])

pred_extended_PS = np.copy(pred_original_PS)
pred_extended_PS[nan_index_PS] = pred_t_PS

# np.save("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/PS_predicted_mass_extended_low_mass_range.npy",
#         pred_extended_PS)
np.save("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/PS_predicted_mass_100_scales_extended_low_mass_range.npy",
        pred_extended_PS)

# ST

pred_original_ST = np.load("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/"
                           "ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
nan_index_ST = np.isnan(pred_original_ST)

pred_t_ST = mp.get_predicted_analytic_mass(m_extended_smoothing, ic, barrier="ST",
                                           trajectories=traj[nan_index_ST, :])

pred_extended_ST = np.copy(pred_original_ST)
pred_extended_ST[nan_index_ST] = pred_t_ST

# np.save("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/ST_predicted_mass_extended_low_mass_range.npy",
#         pred_extended_ST)
np.save("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/ST_predicted_mass_100_scales_extended_low_mass_range.npy",
        pred_extended_ST)

# scp PS_predicted_mass_100_scales_extended_low_mass_range.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/.
# scp ST_predicted_mass_100_scales_extended_low_mass_range.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/.