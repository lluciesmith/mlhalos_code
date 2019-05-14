"""
In this script I am predicting masses empirically according to PS and ST,
adding long-wavelength modes to modify mean density.

Take N realisations of the k=0 mode and for each realisation modify
density contrasts and predict masses.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from scripts.hmf import predict_masses as mp
from scripts.hmf.super_sampling import super_sampling as ssc

ic = parameters.InitialConditionsParameters()
m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
m = m_bins[::2]

den_con = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")
den_con_i = den_con[:, ::2]
del den_con

std_50 = 0.26461567993066265
lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)
std_50_z_99 = std_50 * lin_growth
rho_mean = np.random.normal(0, std_50_z_99)

tr_new = den_con_i + rho_mean
del den_con_i


pred_mass_PS = mp.get_predicted_analytic_mass(m, ic, barrier="spherical",  trajectories=tr_new)
pred_mass_ST = mp.get_predicted_analytic_mass(m, ic, barrier="ST", trajectories=tr_new)

np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/test/PS_all_new_rho.npy",
        pred_mass_PS)
np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/test/ST_all_new_rho.npy",
        pred_mass_ST)
