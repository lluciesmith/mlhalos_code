import numpy as np
import pynbody
from mlhalos import parameters
from scripts.hmf import hmf_simulation as hmf_sim
from scripts.hmf import predict_masses as mp

path = "/Users/lls/Documents/CODE"


def test_number_halos_true_sim():
    initial_parameters = parameters.InitialConditionsParameters(path=path)
    log_bins = np.arange(10, 15, 0.1)

    m, n = hmf_sim.get_true_number_halos_per_mass_bins(initial_parameters, log_bins)
    m1, n1 = hmf_sim.get_simulation_number_of_halos_from_particles_halo_mass(initial_parameters, log_m_bins=log_bins)

    np.testing.assert_allclose(n, n1, rtol=1e-1)
    np.testing.assert_allclose(m, m1, rtol=1e-1)
    print("Passed test")


def test_total_mass_simulation():
    initial_parameters = parameters.InitialConditionsParameters(path=path)
    log_bins = np.arange(10, 15, 0.1)

    m, n = hmf_sim.get_true_total_mass_halos_per_mass_bins(initial_parameters, log_bins)
    m1, n1 = hmf_sim.get_simulation_total_mass(initial_parameters, log_m_bins=log_bins)

    np.testing.assert_allclose(n, n1, rtol=1e-1)
    np.testing.assert_allclose(m, m1, rtol=1e-1)
    print("Passed test")


def test_mena_density_effect():
    initial_parameters = parameters.InitialConditionsParameters(path=path)
    mean_density_box = initial_parameters.mean_density

    mean_rho_pynbody = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc**-3")

    ratio = mean_rho_pynbody/mean_density_box

    traj = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")
    traj_pynbody_mean = traj/ratio

    m_bins = 10 ** np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
    m_bins.units = "Msol h^-1"
    pred_mass_PS_pynbody = mp.get_predicted_analytic_mass(m_bins, initial_parameters, barrier="spherical",
                                                          cosmology="WMAP5", trajectories=traj_pynbody_mean)

    np.save("/Users/lls/Documents/CODE/stored_files/hmf/pynbody_rho_bar/ALL_PS_predicted_masses_1500_even_log_m_spaced"
            ".npy",
            pred_mass_PS_pynbody)
    del pred_mass_PS_pynbody

    pred_mass_ST_pynbody = mp.get_predicted_analytic_mass(m_bins, initial_parameters, barrier="ST", cosmology="WMAP5",
                                                          trajectories=traj_pynbody_mean)

    np.save("/Users/lls/Documents/CODE/stored_files/hmf/pynbody_rho_bar/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy",
        pred_mass_ST_pynbody)

