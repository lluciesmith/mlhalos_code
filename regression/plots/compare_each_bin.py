import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
import matplotlib.pyplot as plt
from regression.plots import plotting_functions as pf


y_den_z0 = np.load("/Users/lls/Documents/mlhalos_files/z0_test/predicted_log_halo_mass.npy")
y_fixed_den = np.load("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/predicted_log_halo_mass.npy")
x = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/true_mass_test_set.npy")

bins_plotting = np.linspace(x.min(), x.max(), 15, endpoint=True)


bin_min = 2
bin_max = 3
ind = np.where((x >= bins_plotting[bin_min]) & (x < bins_plotting[bin_max]))[0]

# compare distributions

plt.figure()
plt.axvline(x=bins_plotting[bin_max], c="k", ls="--", lw=1.5)
plt.axvline(x=bins_plotting[bin_min], c="k", ls="--", lw=1.5)
n0, b0, p0 = plt.hist(y_den_z0[ind], bins=100, histtype="step", color="g", label="ics + $z=0$", lw=1.5)
nden, bden, pden = plt.hist(y_fixed_den[ind], bins=b0, histtype="step", color="r", label="ics only", lw=1.5)
per_sigma_0, per_sigma1 = np.percentile(y_fixed_den[ind], [16, 84])
plt.plot([per_sigma_0, per_sigma1], [4000, 4000], color="r", lw=1.5)
plt.scatter(np.median(y_fixed_den[ind]), 4000, color="r", s=10)

perc_den0, perc_den1 = np.percentile(y_den_z0[ind], [16, 84])
plt.plot([perc_den0, perc_den1], [3000, 3000], color="g", lw=1.5)
plt.scatter(np.median(y_den_z0[ind]), 3000, color="g", s=10)
plt.legend(loc="best")
plt.xlabel(r"$\log (M /\mathrm{M}_{\odot})$", size=17)


# difference in distributions

plt.plot((bden[1:] + bden[:-1])/2, n0 - nden)
plt.xlabel(r"$\log (M /\mathrm{M}_{\odot})$", size=17)
plt.ylabel("N(ics + $z=0$) - N (ics)", size=17)

# 2d histograms

pf.compare_2d_histograms(x[ind], y_den_z0[ind], x[ind], y_fixed_den[ind], title2="ics density",
                         title1="ics + $z=0$ density", x_max=x[ind].max(), x_min=x[ind].min(),
                         y_max=13, y_min=y_fixed_den[ind].min()
                         )

pf.compare_2d_histograms(x, y_den_z0, x, y_fixed_den, title2="ics density",
                         title1="ics + $z=0$ density")