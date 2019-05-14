import numpy as np

path = "/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/features/"

traj = np.lib.format.open_memmap(path + "density_features.npy")
ell = np.lib.format.open_memmap(path + "ellipticity_features.npy")

subset_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/subset_ids.npy")

prol = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/features/prolateness.npy")


corr = np.corrcoef(np.vstack((traj[subset_ids].transpose(), ell[subset_ids].transpose())))