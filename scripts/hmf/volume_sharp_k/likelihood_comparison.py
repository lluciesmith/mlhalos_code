import scripts.hmf.hmf_theory
from mlhalos import distinct_colours
from mlhalos import parameters
import numpy as np
import matplotlib.pyplot as plt
from scripts.hmf import halo_mass as hm
from scripts.hmf import likelihood as lh
from mlhalos import window


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
bins = np.arange(10, 15, 0.1)

# Press-Schechter

m_th, num_th = scripts.hmf.hmf_theory.theoretical_number_halos(ic, kernel="PS")


# original 1500
pred_original_PS = np.load("/share/data1/lls/trajectories_sharp_k/volume_sharp_k/"
                           "ALL_predicted_masses_1500_even_log_m_spaced.npy")


# original 250
tr_sk_vol = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sharp_k.npy")
w_sk = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250, volume="sharp-k")

m_sk_vol_h = np.load("/Users/lls/Documents/CODE/stored_files/hmf/"
                     "250_traj_all_sharp_k_filter_volume_sharp_k_ST_predicted_masses.npy")
hmf_sk = hm.get_empirical_number_halos(m_sk_vol_h, ic)