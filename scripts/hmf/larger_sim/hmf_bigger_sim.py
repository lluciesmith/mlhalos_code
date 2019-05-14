"""
In this script I am testing the hypothesis that my standard simulation
cannot resolve large mass scales and therefore cannot well describe the
upcrossing of trajectories at those scales.

I am therefore using a larger simulation box i.e.,
- grid points 512^3
- L = 200 Mpc h^-1 a
- M_p = 8 times more massive than usual




predicting masses empirically according to PS and ST,
having changed the critical overdensity of spherical collapse assuming the
CORRECT growth factor and not just D(a) = a like in an EdS Universe.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import density
import pynbody
from scripts.hmf import hmf_tests as ht
from multiprocessing import Pool
import gc
from scripts.hmf import predict_masses as mp


if __name__ == "__main__":
    initial_snapshot = "/home/app/scratch/sim200.gadget3"
    ic = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)

    # a = np.load("/Users/lls/Documents/CODE/stored_files/Fourier_transform_matrix_shape_512.npy")
    # L = 200
    # k = 2*np.pi*a/L
    #
    # k_min = k[0,0,1]
    # k_max = k[256,256,256]
    #
    # r_max = 1/k_min
    # r_min = 1/k_max
    #
    # th = pynbody.analysis.hmf.TophatFilter(ic.initial_conditions)
    # m_min = th.R_to_M(r_min)  # This is below 10^10, we don' care so just use radius corresponding to 10^10 Msol h^-1
    # m_max = th.R_to_M(r_max)  # This is 1.046 10^16 Msol h^-1

    m_bins = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
    m_bins.units = "Msol h^-1"

    number_of_filtering = 60
    #
    # r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
    # assert r.units == 'Mpc a h**-1'
    # r_physical = r * ic.initial_conditions.properties['a'] / ic.initial_conditions.properties['h']
    # r_physical.units = "Mpc"
    #
    # r_smoothing = np.array(r_physical)
    #
    #
    # ########## Calculate density contrasts and save in chunks of 60 #########
    #
    # den = density.Density(initial_parameters=ic, window_function="sharp k")
    #
    # # smoothed_density = den.get_smooth_density_for_radii_list(ic, r_smoothing)
    #
    #
    # def fun_multiprocessing(rad):
    #     gc.collect()
    #     return den.get_smooth_density_real_space(ic, rad)
    #
    # for j in range(31):
    #     if j == 30:
    #         r_smoothing_j = r_smoothing[int(j * number_of_filtering):]
    #     else:
    #         r_smoothing_j = r_smoothing[int(j*number_of_filtering):int((j*number_of_filtering) + number_of_filtering)]
    #
    #     pool = Pool(processes=50)
    #     d_smoothed_mult = pool.map(fun_multiprocessing, r_smoothing_j)
    #     pool.close()
    #     pool.join()
    #
    #     smoothed_density = pynbody.array.SimArray(d_smoothed_mult).transpose()
    #     smoothed_density.units = "Msol Mpc**-3"
    #
    #     if j!= 30:
    #         assert smoothed_density.shape == (len(ic.initial_conditions), len(r_smoothing[:number_of_filtering]))
    #
    #     den_con_j = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
    #
    #     np.save("/home/lls/stored_files/sim200/density_contrasts_" + str(j * number_of_filtering) + "_to_"
    #             + str((j * number_of_filtering) + number_of_filtering) + ".npy",
    #             den_con_j)
    #
    #     del smoothed_density
    #     del den_con_j
    #
    # j = 30
    # r_smoothing_j = r_smoothing[int(j * number_of_filtering):]
    # den_j = den.get_smooth_density_real_space(ic, r_smoothing_j)
    # smoothed_density = pynbody.array.SimArray(den_j).transpose()
    # smoothed_density.units = "Msol Mpc**-3"
    # den_con_j = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
    # np.save("/home/lls/stored_files/sim200/density_contrasts_" + str(j * number_of_filtering) + "_to_"
    #             + str((j * number_of_filtering) + number_of_filtering) + ".npy",
    #             den_con_j)


    ######## Merge all density contrast arrays and then split the trajectories in 1000 chunks. #########

    # m = np.zeros((600 + 600))
    # m[:600] = m_bins[:1200:2]
    # m[600:] = m_bins[1200:1800]
    #
    # all_deltas = np.zeros((512**3, len(m)))
    #
    # for j in range(30):
    #     file = np.load("/home/lls/stored_files/sim200/density_contrasts_" + str(j * number_of_filtering)
    #                    + "_to_" + str((j * number_of_filtering) + number_of_filtering) + ".npy")
    #     # if j == 30:
    #     #     all_deltas[:, int(j * number_of_filtering):] = file
    #
    #     if j <= 20:
    #         all_deltas[:, int(j * (number_of_filtering/2)):int((j * (number_of_filtering/2)) +
    #                                                             (number_of_filtering/2))] = file[:, ::2]
    #     else:
    #         all_deltas[:, int(j*number_of_filtering):int((j*number_of_filtering) + number_of_filtering)] = file
    #
    #     del file

    #m = np.zeros((900))
    m = m_bins[::2]
    #m[600:] = m_bins[1200:1800]

    all_deltas = np.zeros((512**3, len(m)))

    for j in range(30):
        file = np.load("/home/lls/stored_files/sim200/density_contrasts_" + str(j * number_of_filtering)
                       + "_to_" + str((j * number_of_filtering) + number_of_filtering) + ".npy")

        all_deltas[:, int(j * (number_of_filtering/2)):int((j * (number_of_filtering/2)) +
                                                                (number_of_filtering/2))] = file[:, ::2]
        del file

    np.save("/share/data1/lls/sim200/all_deltas.npy", all_deltas)

    print("Done building full array of trajectories")

    split_array = np.array(np.array_split(all_deltas, 1000, axis=0))
    del all_deltas

    for i in range(len(split_array)):
        np.save("/share/data1/lls/sim200/trajectories_ids_" + str(i) + ".npy", split_array[i])


    ######### Predict masses for trajectories ######

    predicted_masses_PS = np.array([])
    predicted_masses_ST = np.array([])

    for i in range(1000):
        traj_i = np.load("/share/data1/lls/sim200/trajectories_ids_" + str(i) + ".npy")

        pred_mass_PS_i = mp.get_predicted_analytic_mass(m, ic, barrier="spherical", cosmology="WMAP5",
                                                        trajectories=traj_i)

        predicted_masses_PS = np.concatenate((predicted_masses_PS, pred_mass_PS_i))

        pred_mass_ST_i = mp.get_predicted_analytic_mass(m, ic, barrier="ST", cosmology="WMAP5",
                                                        trajectories=traj_i)
        predicted_masses_ST = np.concatenate((predicted_masses_ST, pred_mass_ST_i))
        del traj_i

    np.save("/share/data1/lls/sim200/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy", predicted_masses_PS)
    np.save("/share/data1/lls/sim200/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy", predicted_masses_ST)




# r_smoothing_60 = r_smoothing[:60]
# pool = Pool(processes=50)
# d_smoothed_mult = pool.map(fun_multiprocessing, r_smoothing_60)
# pool.close()
# pool.join()
#
# smoothed_density = pynbody.array.SimArray(d_smoothed_mult).transpose()
# smoothed_density.units = "Msol Mpc**-3"
# assert smoothed_density.shape == (len(ic.initial_conditions), len(r_smoothing[:60]))
#
# np.save("/home/lls/stored_files/sim200/densities_first_60.npy", d_smoothed_mult)

# np.save("/home/lls/stored_files/sim200/densities_1500_even_log_m.npy",
#         smoothed_density)
#
#
# ########## Calculate density contrast #########
#
# den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
# np.save("/home/lls/stored_files/sim200/ALL_trajectories_1500_even_log_m.npy",
#         den_con)
#
# del smoothed_density
#
#
# ########## Predict PS and ST masses #########
#
# pred_mass_PS = mp.get_predicted_analytic_mass(m_bins, ic, barrier="spherical", cosmology="WMAP5",
#                                                trajectories=den_con)
# np.save("/home/lls/stored_files/sim200/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy", pred_mass_PS)
#
# pred_mass_ST = mp.get_predicted_analytic_mass(m_bins, ic, barrier="ST", cosmology="WMAP5",
#                                                trajectories=den_con)
# np.save("/home/lls/stored_files/sim200/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy", pred_mass_ST)


