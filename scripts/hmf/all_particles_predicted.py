import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import scripts.hmf.hmf
from scripts.hmf import hmf_empirical as hmf_emp
from scripts.hmf import hmf_theory as hmf_th
from scripts.hmf import hmf_simulation as hmf_sim
from mlhalos import parameters
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":

    initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

    # analytic

    m_PS, sig, dndlog10_theory_PS = hmf_th.hmf_theory(initial_parameters, kernel="PS", log_M_min=10,
                                                                      log_M_max=15,
                                                                      delta_log_M=0.1)
    m_ST, sig, dndlog10_theory_ST = hmf_th.hmf_theory(initial_parameters, kernel="ST", log_M_min=10,
                                                                      log_M_max=15,
                                                                      delta_log_M=0.1)

    # theory

    log_m = np.arange(10, 15, 0.1)
    mid_mass, true_dndlog10 = hmf_sim.get_true_dndlog10m(initial_parameters, log_m)


    # empirical

    predicted_mass_PS = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k"
                                "/ALL_predicted_masses_1500_even_log_m_spaced.npy")
    predicted_mass_ST = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k"
                                 "/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")
    m_emp, dndlogm = hmf_emp.get_empirical_total_mass(initial_parameters, predicted_mass_PS, log_m_bins=log_m)
    m_emp_ST, dndlogm_ST = hmf_emp.get_empirical_total_mass(initial_parameters, predicted_mass_ST, log_m_bins=log_m)

    # plot

    #plt.errorbar(mid_mass, true_dndlog10, color="k", fmt="o")
    plt.loglog(mid_mass, true_dndlog10, color="k")
    plt.loglog(m_PS, dndlog10_theory_PS, label="PS", color="b", ls="--")
    plt.loglog(m_ST, dndlog10_theory_ST, label="ST", color="g", ls="--")
    plt.loglog(m_emp, dndlogm, label="emp PS", color="b")
    plt.loglog(m_emp_ST, dndlogm_ST , label="emp ST", color="g")
    plt.legend(loc="best")


