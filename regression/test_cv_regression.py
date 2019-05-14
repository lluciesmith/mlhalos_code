import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import numpy as np
from mlhalos import machinelearning as ml

traj = np.load("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/density_trajectories.npy")
halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")

training_ind = np.random.choice(len(traj), 50000)

feat_training = np.column_stack((traj[training_ind], halo_mass[training_ind]))

clf = ml.MLAlgorithm(feat_training, method="regression", split_data_method=None)
print(clf.best_estimator)
print(clf.algorithm.best_params_)
print(clf.algorithm.best_score_)
