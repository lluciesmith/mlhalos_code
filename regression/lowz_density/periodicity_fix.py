import numpy as np
import matplotlib.pyplot as plt
from regression.plots import plotting_functions as pf

if __name__ == "__main__":
    y_fixed_den = np.load("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/predicted_log_halo_mass.npy")
    y_den = np.log10(np.load("/Users/lls/Documents/mlhalos_files/no_periodicity_fix/"
                             "density_only/predicted_halo_mass.npy"))
    x = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/true_mass_test_set.npy")

    # cumulative distribution difference between predicted masses

    mean_y = np.mean([y_fixed_den, y_den], axis=0)
    diff = abs(y_fixed_den - y_den)/mean_y * 100

    plt.figure()
    n, b, p = plt.hist(diff, normed=True, cumulative=True, histtype="step", bins=100)
    plt.xlabel(r"$(M_{\mathrm{period.fix}} - M_{\mathrm{no fix}}) \%$")
    plt.ylabel("CDF")
    plt.savefig("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/cdf_difference_predicted_masses.png")
    plt.clf()

    # Compare 2d histograms of true vs predicted w and w/o periddicity fix

    pf.compare_2d_histograms(x, y_fixed_den, x, y_den, title1="periodicity fix", title2="no period. fix")
    plt.savefig("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/2d_hist.png")

    # Compare violin plots

    b_t = np.linspace(x.min(), x.max(), 15, endpoint=True)
    pf.compare_violin_plots(y_fixed_den, x, y_den, x, b_t, label1="periodicity fix", label2="no period. fix", color2="r")
    plt.savefig("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/violin_plots.png")

    # Which halos do those particles with greatest difference live in?

    plt.hist(x[np.where(diff > 10)[0]], bins=b_t)


    y_fixed_den = np.load("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/predicted_log_halo_mass.npy")
    y_den_z0 = np.load("/Users/lls/Documents/mlhalos_files/z0_test/predicted_log_halo_mass.npy")
    x = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/true_mass_test_set.npy")

    # Compare 2d histograms of true vs predicted w and w/o periddicity fix
    #
    pf.compare_2d_histograms(x, y_den_z0, x, y_fixed_den, title2="ics density", title1="ics + $z=0$ density")


    # Compare violin plots

    b_t = np.linspace(x.min(), x.max(), 15, endpoint=True)
    pf.compare_violin_plots(y_den_z0, x, y_fixed_den, x, b_t, label1="ics + $z=0$ density", label2="ics density",
                            color2="r", color1="g")
    pf.get_violin_plot_single_prediction(b_t, y_den_z0, x, label_distr="$z=0.1$ density")
    plt.savefig("/Users/lls/Documents/mlhalos_files/z01_test/violin_plots.png")
