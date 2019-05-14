import matplotlib.pyplot as plt
import numpy as np

from regression.plots import plotting_functions as pf


# def violin_plots_density_vs_inertia(shear_predicted, shear_true, density_predicted, density_true, bins, path=None,
#                                     label_0="den+inertia", compare="density", saving_name=None, color="#587058"):
#     if compare == "density":
#         label_1 = "den only"
#         c = "r"
#         if saving_name is None:
#             saving_name = "violins_inertia_vs_density.pdf"
#     elif compare == "shear":
#         label_1 = "den+shear"
#         c = "b"
#         if saving_name is None:
#             saving_name = "violins_inertia_vs_shear.pdf"
#
#     elif compare == "inertia":
#         label_1 = "den+inertia"
#         c = "g"
#         if saving_name is None:
#             saving_name = "violins_inertia_vs_den_plus_inertia.pdf"
#
#     else:
#         raise NameError


def violin_plots_density_vs_inertia(shear_predicted, shear_true, density_predicted, density_true, bins, path=None,
                                        label_0="den+inertia", compare="density", saving_name=None, color="#587058",
                                    c="yellow"):
    shear_pred, shear_mean = pf.get_predicted_masses_in_each_true_m_bin(bins, shear_predicted, shear_true,
                                                                        return_mean=False)
    den_pred, den_mean = pf.get_predicted_masses_in_each_true_m_bin(bins, density_predicted, density_true,
                                                                    return_mean=False)

    width_xbins = np.diff(bins)
    xaxis = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(nrows=1, ncols=1)
    vplot1 = axes.violinplot(den_pred, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(c) for b in vplot1['bodies']]
    axes.errorbar(xaxis, den_mean, xerr=width_xbins / 2, color=c, fmt="o", label=compare)

    vplot = axes.violinplot(shear_pred, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color) for b in vplot['bodies']]
    axes.errorbar(xaxis, shear_mean, xerr=width_xbins / 2, color=color, fmt="o", label=label_0)

    axes.plot(bins, bins, color="k")
    axes.set_xlim(bins.min() - 0.1, bins.max() + 0.1)
    axes.set_ylim(bins.min() - 0.1, bins.max() + 0.1)

    axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    axes.legend(loc=2)
    if path is not None:
        plt.savefig(path + saving_name)

if __name__ == "__main__":
    path_inertia = "/Users/lls/Documents/mlhalos_files/regression/inertia/"
    path_shear = "/Users/lls/Documents/mlhalos_files/regression/shear/"
    path_density = "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"

    # inertia

    inertia_log_true_mass = np.load(path_inertia + "true_halo_mass.npy")
    bins_plotting = np.linspace(inertia_log_true_mass.min(), inertia_log_true_mass.max(), 25, endpoint=True)

    den_plus_inertia = np.load(path_inertia + "inertia_plus_den/predicted_halo_mass.npy")
    den_plus_inertia_log = np.log10(den_plus_inertia)

    # pf.get_violin_plot(bins_plotting, all_log_predicted_mass, all_log_true_mass, return_mean=False, label_distr="All")
    # plt.savefig(path + "violins.png")

    # shear

    shear_log_true_mass = np.load(path_shear + "true_halo_mass.npy")

    shear_predicted_mass = np.load(path_shear + "predicted_halo_mass.npy")
    shear_log_predicted_mass = np.log10(shear_predicted_mass)

    # density

    den_true = np.load(path_density + "true_halo_mass.npy")
    den_predicted = np.load(path_density + "predicted_halo_mass.npy")
    log_den_true = np.log10(den_true)
    log_den_predicted = np.log10(den_predicted)

    # violin_plots_density_vs_inertia(den_plus_inertia_log, inertia_log_true_mass, log_den_predicted,
    #                                 log_den_true, bins_plotting, label_0="den+shear+inertia", compare="density",
    #                                 color="grey",
    #                                 path=path_inertia + "inertia_den_shear/",
    #                                 saving_name="inertia_den_shear_vs_den.pdf")
    #
    # plt.clf()
    # violin_plots_density_vs_inertia(den_plus_inertia_log, inertia_log_true_mass, shear_log_predicted_mass,
    #                                 inertia_log_true_mass, bins_plotting, label_0="den+shear+inertia",
    #                                 compare="inertia",
    #                                 color="grey",
    #                                 path=path_inertia + "inertia_den_shear/",
    #                                 saving_name="inertia_den_shear_vs_inertia_den.pdf"
    #                                 )




    ####################### Importances #######################

    # ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
    # w = window.WindowParameters(initial_parameters=ic)
    # f_imp = np.load(path_inertia + "inertia_den_shear/f_imp.npy")
    plot.plot_importances_vs_mass_scale(f_imp, w.smoothing_masses, save=False, yerr=None,label=["Density", "Inertia"],
                                                                                                  path=".",
                                        title=None, width=0.5, log=False, subplots=2, figsize=(6.9, 5.2), frameon=False,
                                        legend_fontsize=None, ecolor="k")
    # plt.savefig(path_inertia + "f_imp.pdf")







