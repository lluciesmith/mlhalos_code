import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
from mlhalos import parameters
from mlhalos import inertia
from multiprocessing import Pool

path = "/share/data2/lls/regression/local_inertia/tensor/"
testing_ids = np.load("/share/data2/lls/regression/local_inertia/tensor/testing_high_mass.npy")


def pool_local_inertia(particle_id):
    li, eigi = In.get_local_inertia_single_id(particle_id, snapshot, r_smoothing, rho)
    np.save(path + "testing/ids/inertia_tensor_particle_" + str(particle_id) + ".npy", li)
    np.save(path + "testing/ids/eigenvalues_particle_" + str(particle_id) + ".npy", eigi)

    # print("Done and saved particle " + str(particle_id))
    return li

ic = parameters.InitialConditionsParameters(load_final=True)
In = inertia.LocalInertia(testing_ids, initial_parameters=ic)

snapshot = ic.initial_conditions
rho = snapshot["rho"]
filtering_scales = In.filt_scales
r_smoothing = In.filter_parameters.smoothing_radii.in_units(snapshot["pos"].units)[filtering_scales]

pool = Pool(processes=40)
li_particles = pool.map(pool_local_inertia, testing_ids)
pool.close()
pool.join()

# path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/"
#
# # testing_ids = np.load(path + "testing/testing_ids.npy")
# halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
#
# n_tot, b_tot = np.histogram(np.log10(halo_mass[halo_mass!=0]), bins=50)
# t_test = np.load("first_try/testing_particles_saved.npy")
# t_train = np.load("first_try/training_particles_saved.npy")
# n_tr, b_tr = np.histogram(np.log10(halo_mass[ t_test]), bins=b_tot)
#
# in_ids = np.where(halo_mass>0)[0]
# halo_mass_in_ids = np.log10(halo_mass[halo_mass>0])
#
# training_ind = []
# for i in range(len(b_tot) - 1):
#     if n_tr[i] < 3000:
#         num_extra = 3000 - n_tr[i]
#         ind_bin = np.where((halo_mass_in_ids >= b_tot[i]) & (halo_mass_in_ids < b_tot[i + 1]))[0]
#         ids_in_mass_bin = in_ids[ind_bin]
#
#         if ids_in_mass_bin.size == 0:
#             print("Pass")
#             pass
#
#         else:
#             remaining_ids = ids_in_mass_bin[~np.in1d(ids_in_mass_bin, t_test) & ~np.in1d(ids_in_mass_bin, t_train)]
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
# np.save(path + "testing/testing_high_mass.npy", training_ind)