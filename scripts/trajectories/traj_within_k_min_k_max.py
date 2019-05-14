"""
Save sharp-k trajectories above k = 2pi/L.

Save the "cut" trajectories as in
`` /share/data1/lls/trajectories_sharp_k/ALL_traj_above_k_min_box.npy``

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import pynbody
import numpy as np
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters


# Original smoothing scales

# ic = parameters.InitialConditionsParameters()
#
# m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
# m_bins.units = "Msol h^-1"
# r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
# k_smoothing = 1/r
#
# boxsize = pynbody.array.SimArray([50])
# k_min = 2*np.pi/boxsize
#
# k_smoothing = 1/r
# index_above_k_min = np.where(k_smoothing >= k_min)[0]
#
# den_con = np.lib.format.open_memmap("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy", mode="r",
#                                     shape=(16777216, 1516))
# den_con_above_k_min = den_con[:, index_above_k_min]
#
# np.save("/share/data1/lls/trajectories_sharp_k/ALL_traj_above_k_min_box.npy",
#         den_con_above_k_min)


# Take the first 100 trajectories so that you can copy them to mac and see them

ran_index = np.random.choice(np.arange(16777216), 1000)

den_con = np.lib.format.open_memmap("/share/data1/lls/trajectories_sharp_k/ALL_traj_above_k_min_box.npy",
                                    mode="r", shape=(16777216, 1277))

np.save("/share/data1/lls/trajectories_sharp_k/traj_example_above_k_min.npy",
        den_con[ran_index, :])

# scp traj_example_above_k_min.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/CODE/stored_files/traj_examples/.