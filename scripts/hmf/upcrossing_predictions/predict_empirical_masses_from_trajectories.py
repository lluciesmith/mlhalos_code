"""
In this script I am predicting masses empirically according to PS and ST,
having changed the critical overdensity of spherical collapse assuming the
CORRECT growth factor and not just D(a) = a like in an EdS Universe.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from scripts.hmf import predict_masses as mp

ic = parameters.InitialConditionsParameters()
m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"


den_con = np.load("/home/lls/stored_files/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")

pred_mass_PS = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", cosmology="WMAP5", trajectories=den_con)

np.save("/home/lls/stored_files/trajectories_sharp_k/correct_growth/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy",
        pred_mass_PS)
del pred_mass_PS

pred_mass_ST = mp.get_predicted_analytic_mass(m_bins, ic, barrier="ST", cosmology="WMAP5", trajectories=den_con)

np.save("/home/lls/stored_files/trajectories_sharp_k/correct_growth/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy",
        pred_mass_ST)
