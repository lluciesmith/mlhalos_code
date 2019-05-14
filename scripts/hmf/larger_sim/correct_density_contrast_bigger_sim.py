import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from multiprocessing import Pool
from scripts.hmf import predict_masses as mp


def wrong_rho_bar(initial_parameters, snapshot=None):
    if snapshot is None:
        snapshot = initial_parameters.initial_conditions

    boxsize = pynbody.array.SimArray([50.], 'Mpc a h**-1')
    boxsize = boxsize.in_units('Mpc', **snapshot.conversion_context())
    volume = boxsize ** 3

    rho_bar = pynbody.array.SimArray(len(snapshot) * snapshot['mass'][0] / volume)
    rho_bar.units = 'Msol Mpc**-3'
    return rho_bar


def get_ratio_wrong_true_rho_bar(wrong_rho_bar, true_rho_bar):
    ratio = wrong_rho_bar/true_rho_bar
    return ratio


def correct_trajectories_for_wrong_rho_bar(trajectories, ratio_wrong_true):
    true_trajectories = trajectories * ratio_wrong_true
    return true_trajectories


def correct_density_contrasts(wrong_trajectories, initial_parameters):
    wrong_mean_density = wrong_rho_bar(initial_parameters)
    ratio = get_ratio_wrong_true_rho_bar(wrong_mean_density, initial_parameters.mean_density)

    correct_trajectories = correct_trajectories_for_wrong_rho_bar(wrong_trajectories, ratio)
    return correct_trajectories


def predict_PS_ST_masses(number):
    traj_i = np.load("/share/data1/lls/sim200/trajectories_ids_" + str(number) + ".npy")
    traj_correct_i = correct_density_contrasts(traj_i, ic)

    pred_mass_PS_i = mp.get_predicted_analytic_mass(m, ic, barrier="spherical", cosmology="WMAP5",
                                                    trajectories=traj_correct_i)
    np.save("/share/data1/lls/sim200/PS/PS_predicted_masses_" + str(number) + ".npy", pred_mass_PS_i)

    pred_mass_ST_i = mp.get_predicted_analytic_mass(m, ic, barrier="ST", cosmology="WMAP5",
                                                    trajectories=traj_correct_i)
    np.save("/share/data1/lls/sim200/ST/ST_predicted_masses_" + str(number) + ".npy", pred_mass_ST_i)


if __name__ == "__main__":
    initial_snapshot = "/share/data1/lls/sim200/simulation/sim200.gadget3"
    ic = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)

    m_bins = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
    m_bins.units = "Msol h^-1"

    m = m_bins[::2]
    number_of_filtering = 60

    pool = Pool(processes=60)
    range_numbers = np.arange(1000)
    d_smoothed_mult = pool.map(predict_PS_ST_masses, range_numbers)
    pool.join()
    pool.close()
