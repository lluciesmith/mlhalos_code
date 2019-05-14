import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
import pynbody
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
from scripts.hmf.larger_sim import correct_density_contrast_bigger_sim as cd
from multiprocessing import Pool
from scripts.ellipsoidal import ellipsoidal_barrier as eb


def predict_ST_masses_sharp_k(number):
    traj_i = np.load("/share/data1/lls/sim200/trajectories/trajectories_ids_" + str(number) + ".npy")
    traj_correct_i = cd.correct_density_contrasts(traj_i, ic)

    pred_mass_PS_i = mp.get_predicted_analytic_mass(r_smoothing, ic, barrier="spherical", cosmology="WMAP5",
                                                    trajectories=traj_correct_i, filter=SK)
    np.save("/share/data1/lls/sim200/growth_001/PS/PS_predicted_radii_" + str(number) + ".npy", pred_mass_PS_i)

    # pred_mass_ST_i = mp.get_predicted_analytic_mass(m_sk_h[mass_scales], ic, barrier="ST", cosmology="WMAP5",
    #                                                trajectories=traj_correct_i[:, mass_scales])
    # np.save("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/ST/ST_predicted_masses_" + str(number) + ".npy",
    # pred_mass_ST_i)



if __name__ == "__main__":

    # CHANGE KWARGS GROWTH TO BE 0.01

    initial_snapshot = "/share/data1/lls/sim200/simulation/standard200.gadget3"
    final_snapshot="/share/data1/lls/sim200/simulation/snapshot_011"
    ic = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)

    m_bins = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
    m_bins.units = "Msol h^-1"

    m = m_bins[::2]
    r_sph = mp.pynbody_m_to_r(m, ic.initial_conditions)
    r_sph = r_sph * ic.initial_conditions.properties['a'] / ic.initial_conditions.properties['h']

    r_smoothing = r_sph

    k_smoothing = 1/r_smoothing
    SK = eb.SharpKFilter(ic.initial_conditions)

    number_of_filtering = 60

    pool = Pool(processes=60)
    range_numbers = np.arange(1000)
    d_smoothed_mult = pool.map(predict_PS_ST_masses_sharp_k, range_numbers)
    pool.join()
    pool.close()

    SK = eb.SharpKFilter(initial_parameters.initial_conditions)
    ST_k_predicted = mp.get_predicted_analytic_mass(k[::-1], initial_parameters, barrier="ST", cosmology="WMAP5",
                                                    trajectories=den_con, filter=SK)