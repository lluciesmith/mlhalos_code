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
for i in range(shape_tens[0]):
    for j in range(shape_tens[1]):
        eigvals[i,j] = np.linalg.eigvals(in_tens[i, j])

traj = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/"
                                 "density_trajectories.npy")
ell = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/density_subtracted_ellipticity.npy")
prol = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities"
                                 "/density_subtracted_prolateness"
                                 ".npy")
subset_ids = np.load(path + "subset_ids.npy")
t_sub = traj[subset_ids, ::2]
ell_sub = ell[subset_ids, ::2]
prol_sub = prol[subset_ids, ::2]

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
mass_subset = np.log10(halo_mass[subset_ids])

eigvals_sorted = np.sort(eigvals, axis=2)[::-1]


corr_delta = np.corrcoef(np.vstack((t_sub.transpose(), mass_subset)))
corr_eig0 = np.corrcoef(np.vstack((eigvals_sorted[:,:,0].transpose(), mass_subset)))
corr_eig1 = np.corrcoef(np.vstack((eigvals_sorted[:,:,1].transpose(), mass_subset)))
corr_eig2 = np.corrcoef(np.vstack((eigvals_sorted[:,:,2].transpose(), mass_subset)))
corr_ell= np.corrcoef(np.vstack((ell_sub.transpose(), mass_subset)))
corr_prol= np.corrcoef(np.vstack((prol_sub.transpose(), mass_subset)))

plt.figure(figsize=(8.5,6))
plt.plot(m_smoothing, corr_delta[-1, :-1], label="delta")
plt.plot(m_smoothing, corr_ell[-1, :-1], label="ell")
plt.plot(m_smoothing, corr_prol[-1, :-1], label="prol")
plt.plot(m_smoothing, corr_eig0[-1, :-1], label="eig0")
plt.plot(m_smoothing, corr_eig1[-1, :-1], label="eig1")
plt.plot(m_smoothing, corr_eig2[-1, :-1], label="eig2")
plt.axhline(y=0, color="k", ls="--")
plt.ylabel("corr coeff")
plt.xlabel("Smoothing mass")
plt.xscale("log")
plt.legend(loc="best")
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/correlation_halo_mass"
            "/corr_with_log_halo_mass.png")




plt.figure()
corr_delta_eigs = np.corrcoef(np.vstack((t_sub.transpose(), eigvals_sorted[:, :, 0].transpose(),
                                         eigvals_sorted[:, :, 1].transpose(), eigvals_sorted[:, :, 2].transpose())))

corr_delta_shear = np.corrcoef(np.vstack((t_sub.transpose(), ell_sub.transpose(), prol_sub.transpose())))
diff0 = corr_eig0[-1, :-1] - np.diag(corr_delta_eigs[:25, 25:50])
diff1 = corr_eig1[-1, :-1] - np.diag(corr_delta_eigs[:25, 50:75])
diff2 = corr_eig2[-1, :-1] - np.diag(corr_delta_eigs[:25, 75:100])
diff_ell = corr_ell[-1, :-1] - np.diag(corr_delta_shear[:25, 25:50])
diff_prol = corr_prol[-1, :-1] - np.diag(corr_delta_shear[:25, 50:75])
plt.plot(m_smoothing, diff0, label="eig0")
plt.plot(m_smoothing, diff1, label="eig1")
plt.plot(m_smoothing, diff2, label="eig2")
plt.plot(m_smoothing, diff_ell, label="ell")
plt.plot(m_smoothing, diff_prol, label="prol")
plt.xscale("log")
plt.ylabel("cc halo mass - cc delta")
plt.legend(loc="best")
plt.xlabel("Smoothing mass")
plt.axhline(y=0, color="k", ls="--")


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
m = [np.sum(ic.halo[i]["mass"]) for i in range(15)]

corr_delta_ell = np.corrcoef(np.vstack((t_sub.transpose(), ell_sub.transpose())))
plt.figure()
im = plt.imshow(corr_delta_ell[:25, 25:50], cmap="magma", extent=[np.log10(m_smoothing.min()),
                                                                  np.log10(m_smoothing.max()),
                                                                  np.log10(m_smoothing.max()),
                                                                  np.log10(m_smoothing.min())])
[plt.axvline(x=np.log10(m[i]), color="k") for i in range(15)]
plt.xlabel("delta")
plt.ylabel("ell")
plt.colorbar()
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/correlation/ell_delta_w_vlines.pdf")

plt.figure()
corr_delta_prol = np.corrcoef(np.vstack((t_sub.transpose(), prol_sub.transpose())))
plt.imshow(corr_delta_prol[:25, 25:50], cmap="magma", extent=[np.log10(m_smoothing.min()),
                                                                  np.log10(m_smoothing.max()),
                                                                  np.log10(m_smoothing.max()),
                                                                  np.log10(m_smoothing.min())])
[plt.axvline(x=np.log10(m[i]), color="k") for i in range(15)]
plt.xlabel("delta")
plt.ylabel("prol")
plt.colorbar()
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/correlation/prol_delta_w_vlines.pdf")

plt.plot(corr_delta_prol)