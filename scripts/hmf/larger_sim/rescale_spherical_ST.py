import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
from scripts.hmf import predict_masses as mp
from mlhalos import parameters
from scripts.hmf.larger_sim import correct_density_contrast_bigger_sim as corr
from multiprocessing import Pool
import pynbody


def predict_ST_masses(number):
    traj_i = np.load("/share/data1/lls/sim200/trajectories_ids_" + str(number) + ".npy")
    traj_correct_i = corr.correct_density_contrasts(traj_i, ic)

    pred_mass_ST_i = mp.get_predicted_analytic_mass(m, ic, barrier="ST", cosmology="WMAP5",
                                                    trajectories=traj_correct_i)
    np.save("/share/data1/lls/trajectories_sharp_k/rescale_spherical_to_sk_prediction/sim200/"
            "ST_predicted_masses_" + str(number) + ".npy", pred_mass_ST_i)


if __name__ == "__main__":
    # volumes = ["small", "large"]
    volumes = ["large"]

    for volume in volumes:
        if volume == "small":
            ic = parameters.InitialConditionsParameters()

            traj = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")

            m_bins_original = 10 ** np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
            m_bins_original.units = "Msol h^-1"

            m_bins_rescaled = m_bins_original * 9* np.pi/2

            pred_mass_rescaled = mp.get_predicted_analytic_mass(m_bins_rescaled, ic, barrier="ST", cosmology="WMAP5",
                                                           trajectories=traj)

            np.save("/share/data1/lls/trajectories_sharp_k/rescale_spherical_to_sk_prediction"
                    "/ST_rescaled_predicted_masses.npy",
                    pred_mass_rescaled)
        else:
            ic = parameters.InitialConditionsParameters(initial_snapshot="/home/app/scratch/sim200.gadget3",
                                                        #final_snapshot="/home/app/scratch/snapshot_011"
                                                        )

            m_bins_original = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
            m_bins_original.units = "Msol h^-1"
            m_bins_rescaled = m_bins_original * 9 * np.pi / 2

            m = m_bins_rescaled[::2]
            number_of_filtering = 60

            pool = Pool(processes=60)
            range_numbers = np.arange(1000)
            d_smoothed_mult = pool.map(predict_ST_masses, range_numbers)
            pool.join()
            pool.close()

    pred_masses_st = [np.load("/share/data1/lls/trajectories_sharp_k/rescale_spherical_to_sk_prediction/sim200/"
                              "ST_predicted_masses_" + str(i) + ".npy") for i in range(1000)]
    all_pred_masses_st = np.concatenate(pred_masses_st)
    np.save("/share/data1/lls/trajectories_sharp_k/rescale_spherical_to_sk_prediction/sim200/ALL_ST_predicted_masses.npy",
            all_pred_masses_st)

