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


den_con = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")

# std_50 = 0.26461567993066265
# lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)
# std_50_z_99 = std_50 * lin_growth
#
# rho_means = np.random.normal(0, std_50_z_99, size=64)

# num_real = 64
#
# std_50 = 0.26461567993066265
# delta_background_z0 = np.random.normal(0, std_50, size=num_real)
# lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)
#
# pred_mass_PS = np.zeros((num_real, den_con.shape[0]))
# pred_mass_ST = np.zeros((num_real, den_con.shape[0]))
#
# for i in range(num_real):
#     delta_0 = delta_background_z0[i]
#
#     delta_99 = lin_growth * delta_0
#     tr_new = den_con + delta_99
#     # tr_new = den_con * (1 + delta_mean_i)
#     # tr_new = den_con / (1 + delta_mean_i)
#     delta_sc_0 = 1.686 - delta_0
#
#     pred_mass_PS[i, :] = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical",
#                                                         trajectories=tr_new, delta_sc_0=delta_sc_0)
#     pred_mass_ST[i, :] = mp.get_predicted_analytic_mass(m_bins, ic, barrier="ST",
#                                                         trajectories=tr_new, delta_sc_0=delta_sc_0)
#
#     del tr_new
# #
# #
# # particles_halo_3 = np.random.choice(ic.halo[3]['iord'], 60)
# # den_particles = den_con[particles_halo_3, :]
# # del den_con
# #
# # std_50 = 0.073816597322616057
# # rho_means = np.random.normal(1, std_50, size=64)
# #
# # pred_mass_PS = np.zeros((len(den_particles), len(rho_means)))
# # pred_mass_ST = np.zeros((len(den_particles), len(rho_means)))
# #
# # for i in range(len(den_particles)):
# #     new_traj_particles = ssc.trajectories_with_new_mean(den_particles[i], rho_means)
# #     pred_mass_PS[i, :] = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical",
# #                                                         trajectories=new_traj_particles)
# #     pred_mass_ST[i, :] = mp.get_predicted_analytic_mass(m_bins, ic, barrier="ST",
# #                                                         trajectories=new_traj_particles)
#
# # np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/PS_pred_mass_halo_3_test.npy", pred_mass_PS)
# # np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/ST_pred_mass_halo_3_test.npy", pred_mass_ST)
#
# np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/modify_sc_barrier/PS_pred_mass.npy",
#         pred_mass_PS)
# np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/modify_sc_barrier/ST_pred_mass.npy",
#         pred_mass_ST)


num_real = 64

std_50 = 0.74844616674921927
delta_background_z0 = np.random.normal(0, std_50, size=num_real)
lin_growth = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)

#pred_mass_PS = np.zeros((num_real, den_con.shape[0]))
#pred_mass_ST = np.zeros((num_real, den_con.shape[0]))

for i in range(num_real):
    delta_0 = delta_background_z0[i]

    delta_99 = lin_growth * delta_0
    tr_new = den_con + delta_99

    pred_mass_PS_i = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical",
                                                        trajectories=tr_new)
    pred_mass_ST_i = mp.get_predicted_analytic_mass(m_bins, ic, barrier="ST",
                                                        trajectories=tr_new)

    np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/std_07/PS_pred_mass_" + str(
        delta_99) + ".npy", pred_mass_PS_i)
    np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/std_07/ST_pred_mass" + str(
        delta_99) + ".npy", pred_mass_ST_i)

    del tr_new
    del pred_mass_PS_i
    del pred_mass_ST_i