import numpy as np
import matplotlib.pyplot as plt

path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/tensor/"

in_tens = np.load(path + "inertia_tensor_particles.npy")
shape_tens = in_tens.shape
eigvals = np.zeros(shape_tens[:-1])
for i in range(shape_tens[0]):
    for j in range(shape_tens[1]):
        eigvals[i,j] = np.linalg.eigvals(in_tens[i, j])

traj = np.lib.format.open_memmap("/Users/lls/Documents/mlhalos_files/stored_files/shear/shear_quantities/"
                                 "density_trajectories.npy")
subset_ids = np.load(path + "subset_ids.npy")
t_sub = traj[subset_ids, ::2]

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
mass_subset = halo_mass[subset_ids]

eigvals_sorted = np.sort(eigvals, axis=2)[::-1]

# Correlations eigenvalues with density contrast

path_save = "correlation/"
for n in range(25):
    plt.hist2d(t_sub[:, n], eigvals_sorted[:, n, 0], bins=70)
    plt.xlabel(r"$\delta + 1$")
    plt.ylabel(r"eig $0$")
    plt.savefig(path_save + "eig_0_scale_" + str(n) + ".png")
    plt.clf()
    plt.hist2d(t_sub[:, n], eigvals_sorted[:, n, 1], bins=70)
    plt.xlabel(r"$\delta + 1$")
    plt.ylabel(r"eig $1$")
    plt.savefig(path_save + "eig_1_scale_" + str(n) + ".png")
    plt.clf()
    plt.hist2d(t_sub[:, n], eigvals_sorted[:, n, 2], bins=70)
    plt.xlabel(r"$\delta + 1$")
    plt.ylabel(r"eig $2$")
    plt.savefig(path_save + "eig_2_scale_" + str(n) + ".png")
    plt.clf()

# covariance matrices

f = np.vstack((t_sub.transpose(), eigvals_sorted[:,:,0].transpose(), eigvals_sorted[:,:,1].transpose(),
               eigvals_sorted[:,:,2].transpose(), mass_subset))
corr = np.corrcoef(f)
c_0 = corr[:25, 25:50]
c_1 = corr[:25, 50:75]
c_2 = corr[:25, 75:100]

f_shear = np.vstack((traj[:, ::2].transpose(), ell[:, ::2].transpose(), prol[:,::2].transpose()))
corr_shear = np.corrcoef(f_shear)
plt.imshow(corr_shear[:25, 25:50], cmap='magma')


plt.figure()
plt.imshow(c_0, cmap='magma')
plt.xlabel(r'$\delta + 1$')
plt.ylabel(r"eig $0$")
plt.colorbar()
plt.savefig(path_save + "cov_eig_0.png")
plt.clf()

plt.figure()
plt.imshow(c_1, cmap='magma')
plt.xlabel(r'$\delta + 1$')
plt.ylabel(r"eig $1$")
plt.colorbar()
plt.savefig(path_save + "cov_eig_1.png")
plt.clf()

plt.figure()
plt.imshow(c_2, cmap='magma')
plt.xlabel(r'$\delta + 1$')
plt.ylabel(r"eig $2$")
plt.colorbar()
plt.savefig(path_save + "cov_eig_2.png")
plt.clf()

# correlation functions

correlation_delta_eig0 = np.correlate(t_sub[:,0], eigvals_sorted[:,0,0])





# Correlations eigenvalues with halo mass

path_save = "correlation_halo_mass/"
for n in range(25):
    plt.scatter(mass_subset, eigvals_sorted[:, n, 0], alpha=0.1)
    plt.ylabel(r"eig $0$")
    plt.xlabel(r"$M_{\mathrm{halo}}/M_{\odot}$")
    plt.xscale("log")
    plt.savefig(path_save + "eig_0_scale_" + str(n) + ".png")
    plt.clf()
    plt.scatter(mass_subset, eigvals_sorted[:, n, 1], alpha=0.1)
    plt.ylabel(r"eig $1$")
    plt.xlabel(r"$M_{\mathrm{halo}}/M_{\odot}$")
    plt.xscale("log")
    plt.savefig(path_save + "eig_1_scale_" + str(n) + ".png")
    plt.clf()
    plt.scatter(mass_subset, eigvals_sorted[:, n, 2], alpha=0.1)
    plt.ylabel(r"eig $2$")
    plt.xlabel(r"$M_{\mathrm{halo}}/M_{\odot}$")
    plt.xscale("log")
    plt.savefig(path_save + "eig_2_scale_" + str(n) + ".png")
    plt.clf()

int_to_major = np.zeros((6000, 25))
minor_to_major = np.zeros((6000, 25))
for i in range(6000):
    for j in range(25):
        int_to_major[i,j] = eigvals_sorted[i, j, 1]/eigvals_sorted[i, j, 2]
        minor_to_major[i, j] = eigvals_sorted[i, j, 0] / eigvals_sorted[i, j, 2]

path_save = "correlation_halo_mass/ratio_eigs/"
for n in range(25):
    plt.scatter(mass_subset, int_to_major[:, n], alpha=0.1)
    plt.ylabel(r"eig$_2/$eig$_1$")
    plt.xlabel(r"$M_{\mathrm{halo}}/M_{\odot}$")
    plt.xscale("log")
    plt.savefig(path_save + "int_scale_" + str(n) + ".png")
    plt.clf()
    plt.scatter(mass_subset, minor_to_major[:, n], alpha=0.1)
    plt.ylabel(r"eig$_3/$eig$_1$")
    plt.xlabel(r"$M_{\mathrm{halo}}/M_{\odot}$")
    plt.xscale("log")
    plt.savefig(path_save + "minor_scale_" + str(n) + ".png")
    plt.clf()

plt.figure(figsize=(8.5,7))
plt.plot(sm_m[::2], corr[4, 25:50], label=r"$\delta(\mathrm{M_{sm}}=%.2e)$" % sm_m[::2][4])
plt.plot(sm_m[::2], corr[12,25:50], label=r"$\delta(\mathrm{M_{sm}}=%.2e)$" % sm_m[::2][12])
plt.plot(sm_m[::2], corr[14, 25:50], label=r"$\delta(\mathrm{M_{sm}}=%.2e)$" % sm_m[::2][14])
plt.plot(sm_m[::2], corr[16, 25:50], label=r"$\delta(\mathrm{M_{sm}}=%.2e)$" % sm_m[::2][16])
plt.plot(sm_m[::2], corr[22, 25:50], label=r"$\delta(\mathrm{M_{sm}}=%.2e)$" % sm_m[::2][22])
plt.xscale("log")
# plt.ylim(-0.35, 0.05)
plt.legend(loc="best")
plt.xlabel(r"$\mathrm{M_{smoothing}}$")
plt.ylabel(r"Corr$(\delta, \lambda_0)$")
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/inertia/eig0_corr_diff_scales.png")
