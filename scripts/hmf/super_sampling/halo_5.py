import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from scripts.hmf import predict_masses as mp
import pynbody

ic = parameters.InitialConditionsParameters()
ids_halo_5 = np.where(ic.final_snapshot['grp'] == 5)[0]

traj = np.load("/share/data1/lls/trajectories_sharp_k/ALL_traj_1500_even_log_m.npy")
traj_h_5 = traj[ids_halo_5, :]
np.save("/share/data1/lls/trajectories_sharp_k/traj_halo_5.npy", traj_h_5)