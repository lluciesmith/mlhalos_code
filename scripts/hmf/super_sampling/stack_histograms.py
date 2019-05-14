import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from scripts.hmf import halo_mass as hm


ic = parameters.InitialConditionsParameters()
bins = np.arange(10, 15, 0.1)

##### Press-Schechter histograms for number of halos #####

PS = np.load("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/PS_pred_mass_halo_3_test.npy")

n_emp_PS = np.array([hm.get_empirical_number_halos(PS[i][~np.isnan(PS[i])], ic, log_m_bins=bins) for i in range(len(
    PS))])
np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/PS_number_halos_per_realisation.npy", n_emp_PS)

del PS
del n_emp_PS

#### Sheth-Tormen histograms for number of halos #####

ST = np.load("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/ST_pred_mass_halo_3_test.npy")

n_emp_ST = np.array([hm.get_empirical_number_halos(ST[i][~np.isnan(ST[i])], ic, log_m_bins=bins) for i in range(len(
    ST))])
np.save("/share/data1/lls/trajectories_sharp_k/super_sampling/boxsize_50/ST_number_halos_per_realisation.npy", n_emp_ST)

