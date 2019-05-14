import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from scripts.hmf import predict_masses as ht
import pynbody
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from scripts.hmf import predict_masses as mp
from scripts.hmf.larger_sim import correct_density_contrast_bigger_sim as cd
from multiprocessing import Pool


# def predict_ST_radii(number):
#     traj_i = np.load("/share/data1/lls/sim200/trajectories/trajectories_ids_" + str(number) + ".npy")
#     traj_correct_i = cd.correct_density_contrasts(traj_i, ic)
#
#     pred_mass_ST_TH = mp.record_mass_first_upcrossing(b_z, r_smoothing, trajectories=traj_correct_i)
#     np.save("/share/data1/lls/sim200/upcrossings/TH_ST_radii_" + str(number) + ".npy", pred_mass_ST_TH)
#
#     pred_mass_ST_SK = mp.record_mass_first_upcrossing(b_z_SK, r_smoothing, trajectories=traj_correct_i)
#     np.save("/share/data1/lls/sim200/upcrossings/SK_ST_radii_" + str(number) + ".npy", pred_mass_ST_SK)
#
#
# #ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
# ic = parameters.InitialConditionsParameters(initial_snapshot="/share/data1/lls/sim200/simulation/standard200.gadget3",
#                                             final_snapshot="/share/data1/lls/sim200/simulation/snapshot_011",
#                                             load_final=True)
#
# m_bins = 10**np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
# m_bins.units = "Msol h^-1"
# m_bins = m_bins[::2]
# r_smoothing = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
#
#
# # ST barrier with \sigma(R) calculated with top-hat window function
#
# b_z = eb.ellipsoidal_collapse_barrier(r_smoothing, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
#                                  cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=None)
#
#
# # ST barrier with \sigma(R) calculated with sharp-k window function
#
# SK = eb.SharpKFilter(ic.initial_conditions)
# b_z_SK = eb.ellipsoidal_collapse_barrier(r_smoothing, ic, beta=0.485, gamma=0.615, a=0.707, z=99,
#                                  cosmology="WMAP5", output="rho/rho_bar", delta_sc_0=1.686, filter=SK)
#
# pool = Pool(processes=60)
# range_numbers = np.arange(1000)
# d_smoothed_mult = pool.map(predict_ST_radii, range_numbers)
# pool.join()
# pool.close()

# concatenate
a = np.array(np.array_split(np.arange(512**3), 1000, axis=0))

radii_TH = [np.lib.format.open_memmap("/share/data1/lls/sim200/upcrossings/TH_ST_radii_" + str(i) + ".npy", mode="r",
                                            shape=(len(a[i]), 910))
            for i in range(1000)]
radii_TH_all = np.concatenate(radii_TH)
np.save("/share/data1/lls/sim200/upcrossings/predicted_radii_TH_sig.npy", radii_TH_all)

radii_SK = [np.lib.format.open_memmap("/share/data1/lls/sim200/upcrossings/SK_ST_radii_" + str(i) + ".npy", mode="r",
                                            shape=(len(a[i]), 910))
            for i in range(1000)]
radii_SK_all = np.concatenate(radii_SK)
np.save("/share/data1/lls/sim200/upcrossings/predicted_radii_SK_sig.npy", radii_SK_all)
