import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import parameters
from mlhalos import window
import numpy as np
import pynbody

ic = parameters.InitialConditionsParameters()
w = window.WindowParameters(initial_parameters=ic)
r = w.smoothing_radii[-2]

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006/"
                       "training_ids.npy")
ids_between_115_125 = np.where((np.log10(halo_mass[training_ids]) > 11.5) &
                                        (np.log10(halo_mass[training_ids]) <= 12.5))[0]

n = np.random.choice(training_ids[ids_between_115_125], 50000, replace=False)

ic.initial_conditions["rho"].convert_units("Msol Mpc**-3")

t = np.zeros((256**3,))
for particle_id in n:
    pynbody.analysis.halo.center(ic.initial_conditions[particle_id], vel=False)
    ic.initial_conditions.wrap()
    sphere = ic.initial_conditions[pynbody.filt.Sphere(str(r) + " Mpc", (0, 0, 0))]
    t[particle_id] = np.mean(sphere["rho"]) / ic.mean_density

np.save("/home/lls/stored_files/feature_smoothed_real_space.npy", t)
