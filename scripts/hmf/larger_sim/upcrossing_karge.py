import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
from scripts.hmf.larger_sim import correct_density_contrast_bigger_sim as cd
from multiprocessing import Pool


def predict_PS_ST_masses_sharp_k(number):
    traj_i = np.load("/share/data1/lls/sim200/trajectories/trajectories_ids_" + str(number) + ".npy")
    traj_correct_i = cd.correct_density_contrasts(traj_i, ic)

    pred_mass_PS_i = mp.get_predicted_analytic_mass(r_smoothing, ic, barrier="spherical", cosmology="WMAP5",
                                                    trajectories=traj_correct_i, growth=0.01)
    np.save("/share/data1/lls/sim200/growth_001/PS/PS_predicted_radii_" + str(number) + ".npy", pred_mass_PS_i)

    # pred_mass_ST_i = mp.get_predicted_analytic_mass(m_sk_h[mass_scales], ic, barrier="ST", cosmology="WMAP5",
    #                                                trajectories=traj_correct_i[:, mass_scales])
    # np.save("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/ST/ST_predicted_masses_" + str(number) + ".npy",
    # pred_mass_ST_i)

if __name__ == "__main__":

    # CHANGE KWARGS GROWTH TO BE 0.01

    # initial_snapshot = "/share/data1/lls/sim200/simulation/standard200.gadget3"
    # final_snapshot="/share/data1/lls/sim200/simulation/snapshot_011"
    # ic = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)
    #
    # m_bins = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
    # m_bins.units = "Msol h^-1"
    #
    # m = m_bins[::2]
    # r_sph = mp.pynbody_m_to_r(m, ic.initial_conditions)
    # r_sph = r_sph * ic.initial_conditions.properties['a'] / ic.initial_conditions.properties['h']
    #
    # r_smoothing = r_sph
    #
    # # m_sk = ic.mean_density * 6 * np.pi**2 * r_smoothing**3
    # # m_sk.units = "Msol"
    # # m_sk.sim = ic.initial_conditions
    # # m_sk_h = m_sk.in_units("Msol h^-1")
    # #
    # # mass_scales = np.where(m_sk_h <= 2 * 10 ** 15)[0]
    #
    # number_of_filtering = 60
    #
    # pool = Pool(processes=60)
    # range_numbers = np.arange(1000)
    # d_smoothed_mult = pool.map(predict_PS_ST_masses_sharp_k, range_numbers)
    # pool.join()
    # pool.close()

    # concatenate

    pred_masses_ps = [np.load("/share/data1/lls/sim200/growth_001/PS/PS_predicted_radii_" + str(i) + ".npy")
                      for i in range(1000)]
    all_pred_masses_ps = np.concatenate(pred_masses_ps)
    np.save("/share/data1/lls/sim200/growth_001/ALL_PS_predicted_radii.npy", all_pred_masses_ps)

    # pred_masses_st = [np.load("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/ST/ST_predicted_masses_" + str(i) + ".npy") for i in range(1000)]
    # all_pred_masses_st = np.concatenate(pred_masses_st)
    # np.save("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/ALL_ST_predicted_masses.npy", all_pred_masses_st)
