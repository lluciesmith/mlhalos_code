import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import parameters
import numpy as np
from scripts.hmf import mass_predictions_ST as mp
import pynbody

ic = parameters.InitialConditionsParameters()
#particles_in_halos = mp.get_particles_in_halos(ic, halo_min=None)

#
# # d = density.DensityContrasts(initial_parameters=ic, num_filtering_scales=250, path=None, window_function="sharp k",
# #                              volume="sphere")
# #
# # density_contrasts = d.density_contrasts
# # np.save("/home/lls/stored_files/250_traj_all_sharp_k_filter_volume_sphere.npy", density_contrasts)
# #
# # del d
# # del density_contrasts
#
# # d1 = density.DensityContrasts(initial_parameters=ic, num_filtering_scales=1500, path=None, window_function="sharp k",
# #                              volume="sharp-k")
# #
# # density_contrasts1 = d1.density_contrasts
# # np.save("/home/lls/stored_files/1500_traj_all_sharp_k_filter_volume_sharp_k.npy", density_contrasts1)
#
#
# # Get trajectories as a function of smoothing radii which I arbitrarily choose.
#
#
# #smoothing_radii = np.linspace(0.0057291, 0.20, 1500)
#
# smoothing_radii_50 = np.linspace(0.0057291, 0.20, 50)
#
# den_50 = density.Density(initial_parameters=ic, window_function="sharp k")
# smoothed_density_50 = den_50.get_smooth_density_for_radii_list(ic, smoothing_radii_50)
#
# den_con_50 = density.DensityContrasts.get_density_contrast(ic, smoothed_density_50)
# den_con_50_particles_in_halos = den_con_50[particles_in_halos, :]
#
# np.save("/home/lls/stored_files/trajectories_sharp_k/smoothed_density_contrast_50_particles_in_halos.npy",
#         den_con_50_particles_in_halos)
#
# del smoothed_density_50
# del den_50
# del den_con_50
#
# pred_mass_50 = mp.from_radius_predict_mass(smoothing_radii_50, ic, den_con_50_particles_in_halos,
#                                            barrier="spherical")
# np.save("/home/lls/stored_files/trajectories_sharp_k/pred_mass_50_smoothing.npy",
#         pred_mass_50)
#
# del den_con_50_particles_in_halos
# del pred_mass_50
#
# # 250 filtering_radii
#
# smoothing_radii_250 = np.linspace(0.0057291, 0.20, 250)
#
# den_250 = density.Density(initial_parameters=ic, window_function="sharp k")
# smoothed_density_250 = den_250.get_smooth_density_for_radii_list(ic, smoothing_radii_250)
#
# den_con_250= density.DensityContrasts.get_density_contrast(ic, smoothed_density_250)
# den_con_250_particles_in_halos = den_con_250[particles_in_halos, :]
#
# np.save("/home/lls/stored_files/trajectories_sharp_k/smoothed_density_contrast_250_particles_in_halos.npy",
#         den_con_250_particles_in_halos)
#
# del smoothed_density_250
# del den_250
# del den_con_250
#
# pred_mass_250 = mp.from_radius_predict_mass(smoothing_radii_250, ic, den_con_250_particles_in_halos,
#                                             barrier="spherical")
# np.save("/home/lls/stored_files/trajectories_sharp_k/pred_mass_250_smoothing.npy",
#         pred_mass_250)
#
# del den_con_250_particles_in_halos
# del pred_mass_250
#
# # 1500 filtering_radii
#
# smoothing_radii_1500 = np.linspace(0.0057291, 0.20, 1500)
#
# den_1500 = density.Density(initial_parameters=ic, window_function="sharp k")
# smoothed_density_1500 = den_1500.get_smooth_density_for_radii_list(ic, smoothing_radii_1500)
#
#
# den_con_1500 = density.DensityContrasts.get_density_contrast(ic, smoothed_density_1500)
# den_con_1500_particles_in_halos = den_con_1500[particles_in_halos, :]
#
# np.save("/home/lls/stored_files/trajectories_sharp_k/smoothed_density_contrast_1500_particles_in_halos.npy",
#         den_con_1500_particles_in_halos)
#
# del smoothed_density_1500
# del den_1500
# del den_con_1500
#
# pred_mass_1500 = mp.from_radius_predict_mass(smoothing_radii_1500, ic, den_con_1500_particles_in_halos,
#                                              barrier="spherical")
# np.save("/home/lls/stored_files/trajectories_sharp_k/pred_mass_1500_smoothing.npy",
#         pred_mass_1500)
#
# del den_con_1500_particles_in_halos
# del pred_mass_1500


# NOW ALSO DO 1500 EVENLY SPACED IN LOG-MASS

# Log bins in units [Msol h**-1]

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"

# r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
# assert r.units == 'Mpc a h**-1'
# r_physical = r * ic.initial_conditions.properties['a'] / ic.initial_conditions.properties['h']
# r_physical.units = "Mpc"
#
# r_smoothing = np.array(r_physical)
#
# den = density.Density(initial_parameters=ic, window_function="sharp k")
# smoothed_density = den.get_smooth_density_for_radii_list(ic, r_smoothing)
#
# den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
#
# np.save("/home/lls/stored_files/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy",
#         den_con)
#
# del smoothed_density

den_con = np.load("/home/lls/stored_files/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")

# pred_mass_all = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", cosmology="WMAP5",
#                                                trajectories=den_con)
#
# np.save("/home/lls/stored_files/trajectories_sharp_k/ALL_predicted_masses_1500_even_log_m_spaced.npy",
#         pred_mass_all)

pred_mass_all = mp.get_predicted_analytic_mass(m_bins, ic, barrier="ST", cosmology="WMAP5",
                                               trajectories=den_con)

np.save("/home/lls/stored_files/trajectories_sharp_k/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy",
        pred_mass_all)





#
# rho_M = pynbody.array.SimArray(pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol Mpc**-3"))
# rho_M.units = "Msol Mpc**-3"
# den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density, rho_bar=rho_M)
#
# np.save("/home/lls/stored_files/sharp_trajectories_1500_radii_rho_bar_pynbody.npy", den_con)
#
# del smoothed_density
# Predict masses from sharp-k trajectories using a top-hat in real space as a mass-assignment scheme

#traj = np.load("/home/lls/stored_files/sharp_trajectories_1500_radii.npy")

#w = window.WindowParameters(initial_parameters=ic, volume="sphere")
#filtering_masses = w.get_mass_from_radius(ic, smoothing_radii, ic.mean_density)

# smoothing_r_comoving = hmf_tests.radius_comoving_h_units(smoothing_radii, ic.initial_conditions)
# smoothing_m = hmf_tests.pynbody_r_to_m(smoothing_r_comoving, ic.initial_conditions)
#
# den_con = np.load("/home/lls/stored_files/sharp_trajectories_1500_radii_rho_bar_pynbody.npy")
#
# predicted_masses = mp.get_predicted_analytic_mass(smoothing_m, ic, barrier="spherical", cosmology="WMAP5",
#                                                   trajectories=den_con)
#
# np.save("/home/lls/stored_files/sharp_trajectories_1500_PS_predicted_masses_Msol_h.npy", predicted_masses)

