import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import density
import pynbody
from scripts.hmf.larger_sim import subbox as sb
from scripts.hmf.larger_sim import correct_density_contrast_bigger_sim as corr
from multiprocessing import Pool


initial_snapshot = "/home/app/scratch/sim200.gadget3"
# final_snapshot = "/home/app/scratch/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot)

rhoM = pynbody.analysis.cosmology.rho_M(ic_200.initial_conditions, unit="Msol kpc**-3")
# rhoM = ic_200.mean_density
sim = ic_200.initial_conditions

particle = 62155961
length_subbox = 50 * 0.01 / 0.701 * 10 ** 3

ids_subbox = sb.get_ids_subbox_centered_on_particle(sim, particle, length_subbox)
den_subbox = np.mean(sim[ids_subbox]['rho'])
delta_subbox = (den_subbox - rhoM) / rhoM

a = np.array(np.array_split(np.arange(512**3), 1000, axis=0))

num_array = []
indices_num_array = []
for i in range(1000):
    c = np.where(np.in1d(a[i],ids_subbox))[0]
    if len(c) == 0:
        pass
    else:
        num_array.append(i)
        indices_num_array.append(c)
num = np.array(num_array)
indices_num_array = np.array(indices_num_array)


for i in range(len(num)):
    number = num[i]
    indices = indices_num_array[i]
    shape_0 = len(a[number])
    traj_i = np.lib.format.open_memmap("/share/data1/lls/sim200/trajectories_ids_" + str(number) + ".npy", mode="r",
                                       shape=(shape_0, 910))

    traj_ids = traj_i[indices, :]
    del traj_i
    traj_ids_corrected = corr.correct_density_contrasts(traj_ids, ic_200)
    np.save("/share/data1/lls/sim200/subboxes/subbox_62155961/trajectories_ids_" + str(i) + ".npy", traj_ids_corrected)
    del traj_ids_corrected
