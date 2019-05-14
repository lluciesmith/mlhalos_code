import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
from mlhalos import parameters
from mlhalos import window
from mlhalos import density
import pynbody
from regression.adaboost import gbm_04_only as gbm_fun
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# path="/Users/lls/Documents/mlhalos_files/"
saving_path_traj = "/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj_smoothed_above_1e15Msol/"

ic = parameters.InitialConditionsParameters()
w = window.WindowParameters(initial_parameters=ic)
d = density.DensityContrasts(initial_parameters=ic)

m = np.linspace(np.log10(3e10), np.log10(1e15), 50)
width = np.append(np.diff(m), np.diff(m)[-1])
m_all_p = ic.initial_conditions["mass"].sum()
m1 = np.arange(np.log10(1e15), np.log10(m_all_p*5000), step=width[-1])[1:]

M = pynbody.array.SimArray(10**m1)
M.units = "Msol"
r_smoothing = w.get_smoothing_radius_corresponding_to_filtering_mass(ic, M)

den = d.get_smooth_density_for_radii_list(ic, r_smoothing)
traj_high_smoothing = den/ic.mean_density
np.save(saving_path_traj + "density_contrasts_smoothed_above_1e15Msol.npy", traj_high_smoothing)



############## train GBT ##########

# saving_path_traj = "/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/" \
#                    "ic_traj_smoothed_above_1e15Msol"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

all_traj = np.column_stack((traj, traj_high_smoothing))
training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006"
                       "/training_ids.npy")

all_traj_training = all_traj[training_ids]
log_halo_training = np.log10(halo_mass[training_ids])
training_features = np.column_stack((all_traj_training, log_halo_training))


param_grid = {"loss": "lad", "learning_rate": 0.06, "n_estimators": 2000,  "max_depth": 5,  "max_features":"sqrt"}
clf = gbm_fun.train_gbm(training_features, param_grid=param_grid, cv=False)
joblib.dump(clf, saving_path_traj + "clf.pkl")
np.save(saving_path_traj + "importances.npy", clf.feature_importances_)

testing_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006"
                      "/testing_ids.npy")

testing_features = all_traj[testing_ids]
pred = clf.predict(testing_features)
log_halo_testing = np.log10(halo_mass[testing_ids])
np.save(saving_path_traj + "predictions_for_high_smoothing_scales.npy", pred)


# plot importances

# m = np.linspace(np.log10(3e10), np.log10(1e15), 50)
# width = np.append(np.diff(m), np.diff(m)[-1])
# m_all_p = ic.initial_conditions["mass"].sum()
# m1 = np.arange(np.log10(1e15), np.log10(m_all_p*5000), step=width[-1])
#
# m_all = 10**np.concatenate((m, m1))
# m_all = pynbody.array.SimArray(m_all)
# m_all.units = "Msol"
# r_all = w.get_smoothing_radius_corresponding_to_filtering_mass(ic, m_all)
#
# plt.bar(m_all, clf.feature_importances_, label="densities", color="g",alpha=0.7, width=width*2/3, align="center")
# plt.xscale("log")

# test



