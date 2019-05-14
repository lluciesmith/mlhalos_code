import numpy as np
import matplotlib.pyplot as plt
from mlhalos import parameters
from mlhalos import window


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
in_ids = np.where(halo_mass>0)[0]

path_lowz = "/Users/lls/Documents/mlhalos_files/regression/lowz_density/"
lowz_den = np.lib.format.open_memmap(path_lowz + "z8_density_contrasts.npy")
ics_den = np.lib.format.open_memmap(path_lowz + "ics_density_contrasts.npy")

lowz_den_in_ids = lowz_den[in_ids]
ics_in_ids = ics_den[in_ids]
h_m_in_ids = halo_mass[in_ids]

f_1 = np.vstack((lowz_den_in_ids.transpose(), np.log10(h_m_in_ids)))
corr = np.corrcoef(f_1)

f_2 = np.vstack((ics_in_ids.transpose(), np.log10(h_m_in_ids)))
corr_ics = np.corrcoef(f_2)

f_3 = np.vstack((ics_in_ids.transpose(), lowz_den_in_ids.transpose()))
corr_dens = np.corrcoef(f_3)

x_min = np.log10(w.smoothing_masses).min()
x_max = np.log10(w.smoothing_masses).max()
plt.imshow(corr_dens[:50, 50:], cmap='magma', extent=[x_min, x_max, x_max, x_min])
plt.xlabel("ICs density")
plt.ylabel(r"$z=8.89$ density")
plt.colorbar()
plt.savefig("/Users/lls/Desktop/corr_densities.png")
plt.clf()


plt.plot(w.smoothing_masses, corr[-1, :-1], label=r"$z=8.89$")
plt.plot(w.smoothing_masses, corr_ics[-1, :-1], label=r"$z=99$")
plt.legend(loc="best")
plt.xscale("log")
plt.xlabel("Smoothing mass")
plt.ylabel("corr coeff")
plt.savefig("/Users/lls/Desktop/corr_densities_log_halo_mass.png")
