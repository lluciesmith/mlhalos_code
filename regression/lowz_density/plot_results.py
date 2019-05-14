import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
import matplotlib.pyplot as plt
from regression.plots import plotting_functions as pf

y_lowz = np.load("/Users/lls/Documents/mlhalos_files/z0_test/z0_only/predicted_log_halo_mass.npy")
# y_lowz = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/predicted_log_halo_mass.npy")
y_fixed_den = np.load("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/predicted_log_halo_mass.npy")
x = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/true_mass_test_set.npy")

bins_plotting = np.linspace(x.min(), x.max(), 15, endpoint=True)

# 2D HISTOGRAM

pf.compare_2d_histograms(x, y_lowz, x, y_fixed_den,
                         title1="ics + low-z density", title2="ics density", save_path=None)

# VIOLINS

pf.compare_violin_plots(y_lowz, x, y_fixed_den, x,
                        bins_plotting, label1="ics + low-z density", label2="ics density", color1="g", color2="r")
pf.compare_violin_plots(y_lowz, x, y_fixed_den, x,
                        bins_plotting, label1="$z=0$ density", label2="ics density", color1="g", color2="r")

# 2D histograms of low/mid/high mass halos

high_mass = np.where(x>=13)[0]
mid_mass = np.where((x<13) & (x>12))[0]
low_mass = np.where(x<=12)[0]

pf.compare_2d_histograms(x[high_mass], y_lowz[high_mass], x[high_mass], y_fixed_den[high_mass],
                         title1="ics + low-z density", title2="ics density", save_path=None)
pf.compare_2d_histograms(x, y_lowz, x, y_fixed_den,
                         title1="ics + low-z density", title2="ics density", save_path=None)
pf.compare_2d_histograms(x, y_lowz, x, y_fixed_den,
                         title1="ics + low-z density", title2="ics density", save_path=None)