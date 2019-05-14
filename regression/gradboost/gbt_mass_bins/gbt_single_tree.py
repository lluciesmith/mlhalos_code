import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# saving_path = "/share/data2/lls/regression/gradboost/halos_range_115_125/no_feature_above_1e14/"
saving_path = "/share/data2/lls/regression/gradboost/halos_range_115_125/gaussian_smoothing/"
halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")

#### TRY WITH TRAJECTORIES SMOOTHED WITH GAUSSIAN SMOOTHING
# traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/den_con_gaussian_smoothing.npy")

training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "training_ids.npy")
testing_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "testing_ids.npy")

ids_between_115_125 = np.where((np.log10(halo_mass[training_ids]) > 11.5) &
                                        (np.log10(halo_mass[training_ids]) <= 12.5))[0]

### ignore last feature due to sampling of window function in fourier space

features_training = traj[training_ids[ids_between_115_125], :-1]
truth_training = np.log10(halo_mass[training_ids[ids_between_115_125]])

ids_testing_between_115_125  = np.where((np.log10(halo_mass[testing_ids]) > 11.5) &
                                        (np.log10(halo_mass[testing_ids]) <= 12.5))[0]
features_testing = traj[testing_ids[ids_testing_between_115_125], :-1]
truth_testing = np.log10(halo_mass[testing_ids[ids_testing_between_115_125]])