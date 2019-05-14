import numpy as np
import matplotlib.pyplot as plt
from mlhalos import parameters
from mlhalos import window


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
w = window.WindowParameters(initial_parameters=ic)
m_smoothing = w.smoothing_masses[::2]

path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/"

in_tens = np.load(path + "inertia_tensor_particles.npy")
shape_tens = in_tens.shape
eigvals = np.zeros(shape_tens[:-1])
eigavals_subtracted = np.zeros(shape_tens[:-1])

for i in range(shape_tens[0]):
    for j in range(shape_tens[1]):
        eigvalsij= np.linalg.eigvals(in_tens[i, j])
        eigvalsij_sorted = np.sort(eigvalsij)[::-1]
        eigvals[i, j] = eigvalsij_sorted
        eigavals_subtracted[i,j] = eigvalsij_sorted - (np.sum(eigvalsij)/3)

traj = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/"
                                 "density_trajectories.npy")
ell = np.lib.format.open_memmap(
    "/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/density_subtracted_ellipticity.npy")
prol = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities"
                                 "/density_subtracted_prolateness"
                                 ".npy")
subset_ids = np.load(path + "subset_ids.npy")
t_sub = traj[subset_ids, ::2]
ell_sub = ell[subset_ids, ::2]
prol_sub = prol[subset_ids, ::2]

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
mass_subset = np.log10(halo_mass[subset_ids])

corr_delta = np.corrcoef(np.vstack((t_sub.transpose(), mass_subset)))
corr_eig0 = np.corrcoef(np.vstack((eigavals_subtracted[:, :, 0].transpose(), mass_subset)))
corr_eig1 = np.corrcoef(np.vstack((eigavals_subtracted[:, :, 1].transpose(), mass_subset)))
corr_eig2 = np.corrcoef(np.vstack((eigavals_subtracted[:, :, 2].transpose(), mass_subset)))
corr_ell = np.corrcoef(np.vstack((ell_sub.transpose(), mass_subset)))
corr_prol = np.corrcoef(np.vstack((prol_sub.transpose(), mass_subset)))

plt.figure(figsize=(8.5, 6))
plt.plot(m_smoothing, corr_delta[-1, :-1], label="delta")
plt.plot(m_smoothing, corr_ell[-1, :-1], label="ell")
plt.plot(m_smoothing, corr_prol[-1, :-1], label="prol")
plt.plot(m_smoothing, corr_eig0[-1, :-1], label="eig0 (traceless)")
plt.plot(m_smoothing, corr_eig1[-1, :-1], label="eig1 (traceless)")
plt.plot(m_smoothing, corr_eig2[-1, :-1], label="eig2 (traceless)")
plt.axhline(y=0, color="k", ls="--")
plt.ylabel("corr coeff")
plt.xlabel("Smoothing mass")
plt.xscale("log")
plt.legend(loc="best")
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/correlation_halo_mass"
            "/corr_trace_subtracted_with_log_halo_mass.png")