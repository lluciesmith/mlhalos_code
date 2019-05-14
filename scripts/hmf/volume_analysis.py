import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from mlhalos import parameters
from scripts.hmf import predict_masses as mp



# PREDICT MASSES FOR SHARP_K TRAJECTORIES AND V = 6 pi^2 R^3


ic = parameters.InitialConditionsParameters()

m_bins = 10 ** np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"

m = m_bins
r_sph = mp.pynbody_m_to_r(m, ic.initial_conditions)
r_sph = r_sph * ic.initial_conditions.properties['a'] / ic.initial_conditions.properties['h']

r_smoothing = r_sph

m_sk = ic.mean_density * 6 * np.pi**2 * r_smoothing**3
m_sk.units = "Msol"
m_sk.sim = ic.initial_conditions
m_sk_h = m_sk.in_units("Msol h^-1")

# m_sk_h = m_bins*9*np.pi/2

den_con = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")


pred_mass_PS = mp.get_predicted_analytic_mass(m_sk_h, ic, barrier="spherical", cosmology="WMAP5",
                                               trajectories=den_con)

np.save("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/growth_001/ALL_predicted_masses_1500_even_log_m_spaced"
        ".npy",
        pred_mass_PS)

# pred_mass_ST = mp.get_predicted_analytic_mass(m_sk_h, ic, barrier="ST", cosmology="WMAP5",
#                                                trajectories=den_con)
#
# np.save("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy",
#         pred_mass_ST)

