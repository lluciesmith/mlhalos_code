import sys
sys.path.append("/Users/lls/mlhalos_code")
import numpy as np
import matplotlib.pyplot as plt
from mlhalos import parameters
from mlhalos import window
import pynbody


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
in_ids = np.where(halo_mass>0)[0]

den_ics = np.load("/Users/lls/Documents/mlhalos_files/regression/lowz_density/ics_density_contrasts.npy")
den_z8 = np.load("/Users/lls/Documents/mlhalos_files/regression/lowz_density/z8_density_contrasts.npy")
den_nonlin = np.load("/Users/lls/Documents/mlhalos_files/regression/lowz_density/z8_den_subtracted_linear.npy")

D_ics = pynbody.analysis.cosmology.linear_growth_factor(ic.initial_conditions)
z_10 = pynbody.load("/Users/lls/Documents/CODE/Nina-Simulations/double/snapshot_004")
z_10.physical_units()
D_z8 = pynbody.analysis.cosmology.linear_growth_factor(z_10)

f = (D_z8/D_ics) * (den_ics - 1)
z8_feat_no_lin = den_z8 - ( f + 1)
assert np.allclose(z8_feat_no_lin, den_nonlin)


den_z8_in_ids = den_z8[in_ids]
den_nonlin_in_ids = den_nonlin[in_ids]
h_m_in_ids = halo_mass[in_ids]
ics_in_ids = den_ics[in_ids]

f_1 = np.vstack((den_nonlin_in_ids.transpose(), np.log10(h_m_in_ids)))
corr_nonlinear = np.corrcoef(f_1)

f_2 = np.vstack((den_z8_in_ids.transpose(), np.log10(h_m_in_ids)))
corr_z8 = np.corrcoef(f_2)

f_3 = np.vstack((ics_in_ids.transpose(), den_nonlin_in_ids.transpose()))
corr_dens = np.corrcoef(f_3)

x_min = np.log10(w.smoothing_masses).min()
x_max = np.log10(w.smoothing_masses).max()
plt.imshow(corr_dens[:50, 50:], cmap='magma', extent=[x_min, x_max, x_max, x_min])
plt.xlabel("ICs density")
plt.ylabel("low-z density no lin. growth")
plt.colorbar()
# plt.savefig("/Users/lls/Desktop/corr_densities_wo_linear_growth.png")
plt.clf()

plt.figure()
plt.plot(w.smoothing_masses, corr_nonlinear[-1, :-1], label=r"w/o linear growth")
plt.plot(w.smoothing_masses, corr_z8[-1, :-1], label=r"full delta")
plt.legend(loc="best")
plt.xscale("log")
plt.xlabel("Smoothing mass")
plt.ylabel("corr coeff")
# plt.savefig("/Users/lls/Desktop/corr_linear_growth_subtracted.png")