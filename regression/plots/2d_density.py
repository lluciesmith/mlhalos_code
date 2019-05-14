import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import numpy as np
from regression.plots import plotting_functions as pf

# data

path_inertia = "/Users/lls/Documents/mlhalos_files/regression/inertia/"
x = np.load(path_inertia + "true_halo_mass.npy")
y_inden = np.load(path_inertia + "inertia_plus_den/predicted_halo_mass.npy")
y_inden = np.log10(y_inden)

path_density = "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"
y_den = np.load(path_density + "predicted_halo_mass.npy")
y_den = np.log10(y_den)


###### VIOLIN PLOTS WITH BOXES #######

bins_plotting = np.linspace(x.min(), x.max(), 15, endpoint=True)
inertia_pred, inertia_mean = pf.get_predicted_masses_in_each_true_m_bin(bins_plotting, y_inden, x,
                                                                    return_mean=False)
den_pred, den_mean = pf.get_predicted_masses_in_each_true_m_bin(bins_plotting, y_den, x,
                                                                    return_mean=False)
pf.violin_plot_w_percentile(bins_plotting, inertia_pred, den_pred, label=["Density+Inertia", "Density"])
plt.savefig(path_inertia + "inertia_plus_den/boxplot_violins.pdf")
plt.clf()

pf.plot_quantile(bins_plotting, inertia_pred, den_pred, label=["Density+Inertia", "Density"])
plt.savefig(path_inertia + "inertia_plus_den/quantiles.pdf")
plt.clf()

f = np.vstack((t_in.transpose(), eig_in.transpose()))

plt.figure()

corr = np.corrcoef(f) # Toggle between unscaled and scaled
c1 = corr[:50,50:]
plt.imshow(c1, cmap='magma')
plt.xlabel(r'$\delta + 1$')
plt.ylabel(r'$d^2$')
plt.colorbar()



