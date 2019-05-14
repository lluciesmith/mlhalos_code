"""
Restrict the smoothing scales of the sharp-k trajectories such that
the error due to sample variance on the power spectrum is sigma/P(k) < 10%.
This gives a minimum smoothing k-scale.

Save the "cut" trajectories as in
`` /share/data1/lls/trajectories_sharp_k/ALL_traj_above_sample_variance_min_k.npy``

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import pynbody
import numpy as np
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
from scripts.trajectories.cut_sample_variance import sample_variance as sv


# Original smoothing scales

ic = parameters.InitialConditionsParameters()

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
assert r.units == 'Mpc a h**-1'
r.sim = ic.initial_conditions
r_physical = r.in_units("Mpc h**-1")

boxsize = pynbody.array.SimArray([50])
boxsize.units = "Mpc h**-1 a"
boxsize.sim = ic.initial_conditions
boxsize_physical = boxsize.in_units("Mpc h**-1")
k_min = sv.k_scale_sample_variance_power_spectrum(boxsize_physical, accuracy=0.1)

k_smoothing = 1/r_physical
index_above_sample_var = np.where(k_smoothing >= k_min)[0]

r_reliable = r_physical[index_above_sample_var]
assert all(r_reliable <= 1/k_min)

den_con = np.lib.format.open_memmap("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy", mode="r",
                                    shape=(16777216, 1516))
den_con_trustable_k_range = den_con[:, index_above_sample_var]

np.save("/share/data1/lls/trajectories_sharp_k/ALL_traj_above_sample_variance_min_k.npy",
        den_con_trustable_k_range)


# Take the first 100 trajectories so that you can copy them to mac and see them
#
# ran_index = np.random.choice(np.arange(16777216), 1000)
#
# den_con = np.lib.format.open_memmap("/share/data1/lls/trajectories_sharp_k/ALL_traj_above_sample_variance_min_k.npy",
#                                     mode="r", shape=(16777216, 731))
#
# np.save("/share/data1/lls/trajectories_sharp_k/traj_minus_sample_variance_example_2.npy",
#         den_con[ran_index, :])

