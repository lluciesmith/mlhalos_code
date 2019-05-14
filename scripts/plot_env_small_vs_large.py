import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
#sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
import matplotlib.pyplot as plt

d_small = np.load("/Users/lls/Documents/mlhalos_files/stored_files/rho_environment_small_box.npy")
d_large = np.load("/Users/lls/Documents/mlhalos_files/sim200/rho_environment.npy")

min_d = np.array([d_small.min(), d_large.min()]).min()
max_d = np.array([d_small.max(), d_large.max()]).max()


bins = np.logspace(np.log10(min_d), np.log10(max_d), 30, endpoint=True)
mid_bins = (bins[1:] + bins[:-1])

n_small, b = np.histogram(d_small, bins=bins)
n_large, b = np.histogram(d_large, bins=bins)

plt.figure(figsize=(7,7))
plt.bar(mid_bins, n_large/np.sum(n_large), width=np.diff(bins), label=r"L$=200$ Mpc/h", color="r", alpha=0.6)
plt.bar(mid_bins, n_small/np.sum(n_small), width=np.diff(bins), label=r"L$=50$ Mpc/h", color="b", alpha=0.6)
#n_small, b, p = plt.hist(d_small, bins=bins, histtype="step", label=r"L$=50$ Mpc/h", normed=True)
#n_large, b, p = plt.hist(d_large, bins=bins, histtype="step", label=r"L$=200$ Mpc/h", normed=True)
plt.axvline(x=4*10**11, color="k")
plt.legend(loc="best")
plt.xscale("log")
plt.xlabel(r"$\rho [\mathrm{M_\odot}/\mathrm{kpc}^{-3}]$")
# plt.savefig("/Users/lls/Desktop/env_halos_10Mpc.png")

