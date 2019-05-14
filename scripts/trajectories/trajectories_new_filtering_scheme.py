import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import parameters
from mlhalos import density
import numpy as np
from scripts.hmf import mass_predictions_ST as mp

ic = parameters.InitialConditionsParameters()
particles_in_halos = mp.get_particles_in_halos(ic, halo_min=None)

#boundary_schemes=0.038
#bins_low = 120
# saved as /home/lls/stored_files/trajectories_sharp_k/traj_new_filtering.npy

min_r = 0.0057291
max_r = 0.2

boundary_schemes = 0.03
bins_low = 50

smoothing_radii_low = np.linspace(min_r, boundary_schemes, bins_low)
smoothing_radii_high = []
num = boundary_schemes
while num < max_r:
    a = np.log10(num *100)/100
    smoothing_radii_high.append(num + a)
    num = num + a

smoothing_radii = np.concatenate((smoothing_radii_low, smoothing_radii_high))

den = density.Density(initial_parameters=ic, window_function="sharp k")
smoothed_density = den.get_smooth_density_for_radii_list(ic, smoothing_radii)

den_con = density.DensityContrasts.get_density_contrast(ic, smoothed_density)
den_con_particles_in_halos = den_con[particles_in_halos, :]

np.save("/home/lls/stored_files/trajectories_sharp_k/traj_new_filtering_bound_030.npy",
        den_con_particles_in_halos)