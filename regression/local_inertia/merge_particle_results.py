import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
import os
import re


def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)

path = "/share/data2/lls/regression/local_inertia/tensor/"

# TRAINING

a = []
for filename in os.listdir(path + "ids_50scales/"):
    a.append(int(get_numbers_from_filename(filename)))

a = np.unique(a)
np.save(path + "first_try/training_particles_saved.npy", a)

eig = np.zeros((len(a), 50, 3))
for i in range(len(a)):
    eig[i] = np.load(path + "ids_50scales/eigenvalues_particle_" + str(a[i]) + ".npy")
np.save(path + "first_try/training_eigenvalues_particles.npy", eig)

# # TESTING
#
# a = []
# for filename in os.listdir(path + "testing/ids/"):
#     a.append(int(get_numbers_from_filename(filename)))
#
# a = np.unique(a)
# np.save(path + "first_try/testing_particles_saved.npy", a)
#
# eig = np.zeros((len(a), 50, 3))
# for i in range(len(a)):
#     eig[i] = np.load(path + "testing/ids/eigenvalues_particle_" + str(a[i]) + ".npy")
# np.save(path + "first_try/testing_eigenvalues_particles.npy", eig)
# del a

# cd /share/data2/lls/regression/local_inertia/tensor/first_try
# scp training_particles_saved.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/first_try/
# scp testing_particles_saved.npy lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/first_try/
