import sys
sys.path.append("/home/lls/mlhalos_code/")
import re
import os
import numpy as np
from mlhalos import parameters
from mlhalos import inertia
from multiprocessing import Pool


def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)


def pool_local_inertia(particle_id):
    li, eigi = In.get_local_inertia_single_id(particle_id, snapshot, r_smoothing, rho)
    np.save(path + "ids_50scales/inertia_tensor_particle_" + str(particle_id) + ".npy", li)
    np.save(path + "ids_50scales/eigenvalues_particle_" + str(particle_id) + ".npy", eigi)

    # print("Done and saved particle " + str(particle_id))
    return li


path = "/share/data2/lls/regression/local_inertia/tensor/"
# training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
#
# a = []
# for filename in os.listdir(path + "ids_50scales/"):
#     a.append(int(get_numbers_from_filename(filename)))
#
# a = np.array(a)
# ids_remaining = training_ids[~np.in1d(training_ids, a)]
# del a
#ids_remaining = np.load(path + "training_high_mass.npy")
ids_remaining = np.load(path + "ran_above_137.npy")


ic = parameters.InitialConditionsParameters(load_final=True)
#In = inertia.LocalInertia(training_ids, initial_parameters=ic)
In = inertia.LocalInertia(ids_remaining, initial_parameters=ic)

snapshot = ic.initial_conditions
rho = snapshot["rho"]

filtering_scales = In.filt_scales
r_smoothing = In.filter_parameters.smoothing_radii.in_units(snapshot["pos"].units)[filtering_scales]

pool = Pool(processes=40)
li_particles = pool.map(pool_local_inertia, ids_remaining)
pool.close()
pool.join()


# n_tot, b_tot, p_tot = plt.hist(np.log10(halo_mass[halo_mass!=0]), bins=50)
# n_tr, b_tr, p_tr = plt.hist(np.log10(halo_mass[t_train]), bins=b_tot)
# in_ids = np.where(halo_mass>0)[0]
# halo_mass_in_ids = np.log10(halo_mass[halo_mass>0])
# training_ind = []
# for i in range(len(b_tot) - 1):
#     if n_tr[i] < 3000:
#         num_extra = 3000 - n_tr[i]
#         ind_bin = np.where((halo_mass_in_ids >= b_tot[i]) & (halo_mass_in_ids < b_tot[i + 1]))[0]
#         ids_in_mass_bin = in_ids[ind_bin]
#         if ids_in_mass_bin.size == 0:
#             print("Pass")
#             pass
#
#         else:
#             remaining_ids = ids_in_mass_bin[~np.in1d(ids_in_mass_bin, t_train)]
#             ind_correct = np.in1d(in_ids, remaining_ids)
#
#             radii_in_mass_bin = r_fraction[ind_correct]
#             ids_in_mass_bin = in_ids[ind_correct]
#
#             num_p = int(num_extra/4)
#             ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
#             print((num_p / len(ids_in_mass_bin[radii_in_mass_bin < 0.3]) * 100))
#             ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
#                                       replace=False)
#             print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)]) * 100))
#             ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
#                                      replace=False)
#             print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)]) * 100))
#             ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)
#             print((num_p / len(ids_in_mass_bin[radii_in_mass_bin >= 1]) * 100))
#
#             training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
#             training_ind.append(training_ids_in_bin)
#
# training_ind = np.concatenate(training_ind)
# np.save(path + "training/training_high_mass.npy", training_ind)
