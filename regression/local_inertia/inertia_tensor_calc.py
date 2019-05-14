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
    li = In.get_local_inertia_single_id(particle_id, snapshot, r_smoothing, rho)
    np.save(path + "ids/inertia_tensor_particle_" + str(particle_id) + ".npy", li)

    # print("Done and saved particle " + str(particle_id))
    return li


path = "/share/data2/lls/regression/local_inertia/tensor/"
tr_ids = np.load(path + "subset_ids.npy")

ic = parameters.InitialConditionsParameters(load_final=True)
In = inertia.LocalInertia(tr_ids, initial_parameters=ic)

snapshot = ic.initial_conditions
rho = snapshot["rho"]
# rho_mean = (np.sum(snapshot["mass"]) / snapshot.properties["boxsize"] ** 3).in_units("Msol kpc**-3")
# C = rho - rho_mean

filtering_scales = In.filt_scales
r_smoothing = In.filter_parameters.smoothing_radii.in_units(snapshot["pos"].units)[filtering_scales]

ids_remaining = tr_ids[:]
pool = Pool(processes=40)
li_particles = pool.map(pool_local_inertia, ids_remaining)
pool.close()
pool.join()


### Saving stuff after

a = []
for filename in os.listdir(path + "ids/"):
    a.append(int(get_numbers_from_filename(filename)))

l = np.zeros((len(a), 25, 3, 3))
for i in range(len(a)):
    l[i] = np.load(path + "ids/inertia_tensor_particle_" + str(a[i]) + ".npy")
np.save(path + "inertia_tensor_particles.npy", l)


in_tens = np.load("inertia_tensor_particles.npy")
shape_tens = in_tens.shape
eigvals = np.zeros(shape_tens[:-1])
for i in range(shape_tens[0]):
    for j in range(shape_tens[1]):
        eigvals[i,j] = np.linalg.eigvals(in_tens[i, j])


# To generate subset_ids.npy
# halo_mass_in_ids = halo_mass[halo_mass > 0]
#
# # sort ids in halos and corresponding r/r_vir value
#
# radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
# radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")
# fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
# ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
# ind_sorted = np.argsort(ids_in_halo)
#
# ids_in_halo_mass = ids_in_halo[ind_sorted].astype("int")
# r_fraction = fraction[ind_sorted]
# del fraction
# del ids_in_halo
#
#
# Select a balanced training set
# Take particle ids in each halo mass bin

# n, log_bins = np.histogram(np.log10(halo_mass_in_ids), bins=15)
# bins = 10**log_bins
#
# training_ind = []
# for i in range(len(bins) - 1):
#     ind_bin = np.where((halo_mass_in_ids >= bins[i]) & (halo_mass_in_ids < bins[i + 1]))[0]
#     ids_in_mass_bin = ids_in_halo_mass[ind_bin]
#
#     if ids_in_mass_bin.size == 0:
#         print("Pass")
#         pass
#
#     else:
#         if i == 49:
#             num_p = 100
#             # num_p = 4000
#         else:
#             num_p = 100
#             # num_p = 2500
#
#         radii_in_mass_bin = r_fraction[ind_bin]
#
#         ids_03 = np.random.choice(ids_in_mass_bin[radii_in_mass_bin < 0.3], num_p, replace=False)
#         print((num_p/len(ids_in_mass_bin[radii_in_mass_bin < 0.3]) * 100))
#         ids_06 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)], num_p,
#                                   replace=False)
#         print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.3) & (radii_in_mass_bin < 0.6)]) * 100))
#         ids_1 = np.random.choice(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)], num_p,
#                                  replace=False)
#         print((num_p / len(ids_in_mass_bin[(radii_in_mass_bin >= 0.6) & (radii_in_mass_bin < 1)]) * 100))
#         ids_outer = np.random.choice(ids_in_mass_bin[radii_in_mass_bin >= 1], num_p, replace=False)
#         print((num_p / len(ids_in_mass_bin[radii_in_mass_bin >= 1]) * 100))
#
#         training_ids_in_bin = np.concatenate((ids_03, ids_06, ids_1, ids_outer))
#         training_ind.append(training_ids_in_bin)
#
# training_ind = np.concatenate(training_ind)
