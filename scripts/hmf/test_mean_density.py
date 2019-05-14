import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from mlhalos import parameters
from scripts.hmf import predict_masses as mp



initial_parameters = parameters.InitialConditionsParameters()
mean_density_box = initial_parameters.mean_density

mean_rho_pynbody = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc**-3")

ratio = mean_rho_pynbody / mean_density_box

traj = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")
traj_pynbody_mean = traj / ratio

m_bins = 10 ** np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
pred_mass_PS_pynbody = mp.get_predicted_analytic_mass(m_bins, initial_parameters, barrier="spherical",
                                                      cosmology="WMAP5", trajectories=traj_pynbody_mean)

np.save("/share/data1/lls/trajectories_sharp_k/pynbody_rho_mean/ALL_PS_predicted_masses_1500_even_log_m_spaced"
        ".npy",
        pred_mass_PS_pynbody)
del pred_mass_PS_pynbody

pred_mass_ST_pynbody = mp.get_predicted_analytic_mass(m_bins, initial_parameters, barrier="ST", cosmology="WMAP5",
                                                      trajectories=traj_pynbody_mean)

np.save("/share/data1/lls/trajectories_sharp_k/pynbody_rho_mean/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy",
        pred_mass_ST_pynbody)