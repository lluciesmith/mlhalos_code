import matplotlib.pyplot as plt
import numpy as np
from regression.plots import plotting_functions as pf
from mlhalos import plot
from mlhalos import parameters
from mlhalos import window
import seaborn as sns


def violin_plots(predicted1, true1, predicted2, true2, bins, label1="den+inertia", label2="density",
                 path=None, saving_name=None,
                 color1="#587058", color2="yellow"):
    pred1, mean1 = pf.get_predicted_masses_in_each_true_m_bin(bins, predicted1, true1,
                                                                        return_mean=False)
    pred2, mean2 = pf.get_predicted_masses_in_each_true_m_bin(bins, predicted2, true2,
                                                                    return_mean=False)

    width_xbins = np.diff(bins)
    xaxis = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(nrows=1, ncols=1)
    vplot1 = axes.violinplot(pred1, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color1) for b in vplot1['bodies']]
    axes.errorbar(xaxis, mean1, xerr=width_xbins / 2, color=color1, fmt="o", label=label1)

    vplot = axes.violinplot(pred2, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color2) for b in vplot['bodies']]
    axes.errorbar(xaxis, mean2, xerr=width_xbins / 2, color=color2, fmt="o", label=label2)

    axes.plot(bins, bins, color="k")
    axes.set_xlim(bins.min() - 0.1, bins.max() + 0.1)
    axes.set_ylim(bins.min() - 0.1, bins.max() + 0.1)

    axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    axes.legend(loc=2)
    if path is not None:
        plt.savefig(path + saving_name)


if __name__ == "__main__":
    #path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/reduced_training_set/"
    path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/reduced_training_set/ran_5k/"
    #path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/reduced_training_set/old/random_5k/"

    # local inertia + density vs density only

    log_true_mass = np.load(path + "true_test_halo_mass.npy")
    log_true_mass = np.log10(log_true_mass)
    bins_plotting = np.linspace(log_true_mass.min(), log_true_mass.max(), 15, endpoint=True)

    den_plus_local_inertia = np.load(path + "predicted_halo_mass.npy")
    den_plus_inertia_log = np.log10(den_plus_local_inertia)

    den_predicted = np.load(path + "den_only/predicted_halo_mass.npy")
    log_den_predicted = np.log10(den_predicted)

    inertia_only = np.load(path + "../inertia_only/5k_predicted_halo_mass.npy")
    log_inertia_only = np.log10(inertia_only)

    # plt.figure()
    # violin_plots(den_plus_inertia_log, log_true_mass, log_den_predicted, log_true_mass,
    #              bins_plotting, label1="den+local inertia", label2="density", color1="magenta", color2="grey")
    #
    # plt.figure()
    # corner.corner(np.column_stack((log_true_mass, den_plus_inertia_log, log_den_predicted)),
    #               labels=["True mass", "Density+local inertia", "Density"])

    # plt.figure()
    # corner.corner(np.column_stack((np.log10(log_true_mass), log_den_predicted)))

    path_density = "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"
    den_true_old = np.load(path_density + "true_halo_mass.npy")
    den_predicted_old = np.load(path_density + "predicted_halo_mass.npy")
    log_den_true_old = np.log10(den_true_old)
    log_den_predicted_old = np.log10(den_predicted_old)

    ###### VIOLIN PLOTS #####

    plt.figure()
    violin_plots(log_den_predicted_old, log_den_true_old, log_den_predicted, log_true_mass,
                 bins_plotting, label1="density (full box)", label2="density (subset box)", color1="magenta",
                 color2="grey",
                 path=path,
                 saving_name="density_subset_vs_full.png")
    plt.clf()

    plt.figure()
    violin_plots(den_plus_inertia_log, log_true_mass, log_den_predicted, log_true_mass,
                 bins_plotting, label1="local inertia+density", label2="density only", color1="magenta",
                 color2="grey",
                 path=path,
                 saving_name="density_vs_loc_inertia_plus_den.png")

    plt.figure()
    violin_plots(log_inertia_only, log_true_mass, log_den_predicted, log_true_mass,
                 bins_plotting, label1="inertia only", label2="density only", color1="magenta",
                 color2="grey",
                 path=path,
                 saving_name="../inertia_only/density_vs_loc_inertia_only.png"
                 )


    plt.figure()
    violin_plots(log_inertia_only, log_true_mass, den_plus_inertia_log, log_true_mass,
                 bins_plotting, label1="inertia only", label2="density+inertia", color1="magenta",
                 color2="grey",
                 path=path,
                 saving_name="../inertia_only/density_plus_inertia_vs_loc_inertia_only.png"
                 )

    ###### COMPARE PREDICTIONS WITH TRUE MASS #####

    x = log_true_mass
    y_inden = den_plus_inertia_log
    y_den = log_den_predicted

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(11, 5))
    fig.subplots_adjust(hspace=0.3, left=0.07, right=0.93, top=0.94)
    ax = axs[0]
    hb = ax.hexbin(x[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], gridsize=50, cmap='inferno',
                   #bins="log",
                   #vmin=0, vmax=4.5
                   )
    ax.plot(x[(x <= 14) & (x > 11)], x[(x <= 14) & (x > 11)], color="k")
    ax.set_ylim(11, 14)
    ax.set_title("Local Inertia+density")
    ax.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    ax.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r'$N$')

    ax = axs[1]
    hb1 = ax.hexbin(x[(x <= 14) & (x > 11)], y_den[(x <= 14) & (x > 11)], gridsize=50, cmap='inferno',
                   #bins="log",
                   #vmin=0, vmax=4.5
                   )
    ax.plot(x[(x <= 14) & (x > 11)], x[(x <= 14) & (x > 11)], color="k")
    ax.set_ylim(11, 14)
    ax.set_title("Density only")
    ax.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    ax.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r'$N$')
    plt.savefig(path + "2d_hist_vs_true_mass.png")
    plt.clf()

    plt.figure(figsize=(6.9, 5.2))
    g = (sns.jointplot(x[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], kind="kde", stat_func=None))\
        .set_axis_labels("$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", r"$\log (M_\mathrm{predicted}/\mathrm{M}_{"
                                                                        r"\odot})$", size=17)
    g.ax_joint.plot([10.5, 14.5], [10.5, 14.5], "k")
    plt.savefig(path + "dplot_loc_in_plus_den_vs_true.png")
    plt.clf()

    plt.figure(figsize=(6.9, 5.2))
    g = (sns.jointplot(y_den[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], kind="kde", stat_func=None))\
        .set_axis_labels("$\log (M_\mathrm{density}/\mathrm{M}_{\odot})$", r"$\log (M_\mathrm{"
                                                                           r"loc.inertia+density}/\mathrm{M}_{"
                                                                        r"\odot})$", size=17)
    g.ax_joint.plot([10.5, 14.5], [10.5, 14.5], "k")
    plt.savefig(path + "dplot_loc_in_plus_den_vs_den.png")



    ###### COMPARE DENSITY and DENSITY+INERTIA and also DENSITY SUBSET IDS vs DENSITY FULL BOX #####

    ids_tested_ran5k = np.load(path + "ids_tested.npy")
    old_ids_tested = np.load(path_density + "testing_ids.npy")
    ind_old = np.in1d(old_ids_tested, ids_tested_ran5k)
    ind_new = np.in1d(ids_tested_ran5k, old_ids_tested)

    x = log_den_predicted[ind_new]
    y_inden = den_plus_inertia_log[ind_new]
    y_den = log_den_predicted_old[ind_old]

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(11, 5))
    fig.subplots_adjust(hspace=0.3, left=0.07, right=0.93, top=0.94)
    ax = axs[0]
    hb = ax.hexbin(x[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], gridsize=50, cmap='inferno',
                   #bins="log",
                   #vmin=0, vmax=4.5
                   )
    ax.plot(x[(x <= 14) & (x > 11)], x[(x <= 14) & (x > 11)], color="k")
    ax.set_ylim(11, 14)
    # ax.set_title("Inertia+density")
    ax.set_xlabel("Subset box - density only", size=17)
    ax.set_ylabel("Subset box - density+inertia", size=17)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r'$N$')

    ax = axs[1]
    hb = ax.hexbin(x[(x <= 14) & (x > 11)], y_den[(x <= 14) & (x > 11)], gridsize=50, cmap='inferno',
                   #bins="log",
                   #vmin=0, vmax=4.5
                   )
    ax.plot(x[(x <= 14) & (x > 11)], x[(x <= 14) & (x > 11)], color="k")
    ax.set_ylim(11, 14)
    # ax.set_title("Density only")
    ax.set_xlabel("Subset box - density only", size=17)
    ax.set_ylabel("Full box - density only", size=17)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r'$N$')
    plt.savefig(path + "2d_hist_vs_den_only.png")
    plt.clf()

    ####################### Importances #######################

    # ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
    # w = window.WindowParameters(initial_parameters=ic)
    # f_imp = np.load(path + "den_only/f_imp.npy")
    # plot.plot_importances_vs_mass_scale(f_imp, w.smoothing_masses, save=False, yerr=None,
    #                                     label="Density", path=".",
    #                                     title=None, width=0.5, log=False, subplots=1, figsize=(6.9, 5.2), frameon=False,
    #                                     legend_fontsize=None, ecolor="k")
    # plt.savefig(path + "den_only/f_imp_den_only.png")
    # plt.clf()
    #
    # f_imp = np.load(path + "f_imp.npy")
    # plot.plot_importances_vs_mass_scale(f_imp, w.smoothing_masses, save=False, yerr=None,
    #                                     label=["Density", r"Inertia $\lambda_0$", r"Inertia $\lambda_1$",
    #                                            r"Inertia $\lambda_2$"], path=".",
    #                                     title=None, width=0.5, log=False, subplots=4, figsize=(6.9, 5.2), frameon=False,
    #                                     legend_fontsize=None, ecolor="k")
    # plt.savefig(path + "f_imp_loc_inertia_plus_den.png")