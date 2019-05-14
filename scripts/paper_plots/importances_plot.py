from mlhalos import plot
import importlib
importlib.reload(plot)
import matplotlib
from mlhalos import parameters
from mlhalos import window
from mlhalos import distinct_colours
import matplotlib.pyplot as plt
import numpy as np


def plot_importances(density_importance, mass_scales, initial_parameters, figsize=(6.9, 5.2), yerr=None, ecolor="k",
                     label="Density", subplots=1):
    plot.plot_importances_vs_mass_scale(density_importance, mass_scales, yerr=yerr, save=False, label=label,
                                        path=".", title=None, width=0.5, log=False, subplots=subplots, figsize=figsize,
                                        ecolor=ecolor)

    m_IN_max = initial_parameters.halo[initial_parameters.min_halo_number]['mass'].sum()
    m_IN_min = initial_parameters.halo[initial_parameters.max_halo_number]['mass'].sum()

    # IN/OUT BOUNDARY
    #color_boundary = distinct_colours.get_distinct(2)[1]
    color_boundary = "k"
    y_pos = density_importance.max()
    plt.axvline(x=m_IN_min, color=color_boundary, alpha=100, ls="--", lw=3)
    plt.text(10**(np.log10(m_IN_min) + 0.1), y_pos, "IN",
             fontweight="bold",
             fontsize=18, horizontalalignment="left",
           color=color_boundary, alpha=100)
    plt.text(10**(np.log10(m_IN_min) - 0.1), y_pos, "OUT",
             fontweight="bold",
             fontsize=18, horizontalalignment="right",
           color=color_boundary, alpha=100)

    # SHOW LARGEST HALO WITH ARROW

    # y_posarrow = density_importance[(np.abs(mass_scales-m_IN_max)).argmin()] + 0.005
    # plt.annotate("", xy=(m_IN_max, y_posarrow), xycoords='data', xytext=(m_IN_max, y_posarrow*1.5), textcoords='data',
    #              arrowprops=dict(arrowstyle="simple", facecolor='red', edgecolor="k"), horizontalalignment='center',
    #              verticalalignment='center')

    # SHOW LARGEST HALO WITH GREY LINE

    plt.axvline(m_IN_max, color="grey", lw=2)
    plt.annotate("Most massive halo", xy=(m_IN_max, 0.05), xycoords='data', xytext=(m_IN_max - 0.5 * 10**14, 0.05),
                 textcoords='data', horizontalalignment='center',
                 verticalalignment='center', rotation="vertical", color="grey", fontsize=15)


def plot_density_shear_importance(importance, mass_scales, initial_parameters, figsize=(6.9, 5.2), yerr=None,
                                  ecolor="k"):
    fig, ax1, ax2, ax3 = plot.plot_importances_vs_mass_scale(importance, mass_scales, save=False, yerr=yerr,
                                                             label=["Density", "Ellipticity","Prolateness"],
                                                             path=".", title=None, width=0.5, log=False, subplots=3,
                                                             figsize=figsize,  ecolor=ecolor)
    m_IN_max = initial_parameters.halo[initial_parameters.min_halo_number]['mass'].sum()
    m_IN_min = initial_parameters.halo[initial_parameters.max_halo_number]['mass'].sum()

    # IN/OUT BOUNDARY

    y_pos = importance.min() + (importance.min() * 2.2)
    #color_boundary = distinct_colours.get_distinct(5)[4]
    color_boundary = "k"
    ax1.axvline(x=m_IN_min, color=color_boundary, alpha=100, ls="--", lw=3)
    ax2.axvline(x=m_IN_min, color=color_boundary, alpha=100, ls="--", lw=3)
    ax3.axvline(x=m_IN_min, color=color_boundary, alpha=100, ls="--", lw=3)
    ax3.text(10**(np.log10(m_IN_min) + 0.1), y_pos, "IN", fontweight="bold", fontsize=18, horizontalalignment="left",
           color=color_boundary, alpha=100)
    ax3.text(10**(np.log10(m_IN_min) - 0.1), y_pos, "OUT", fontweight="bold",fontsize=18, horizontalalignment="right",
           color=color_boundary, alpha=100)

    # SHOW LARGEST HALO WITH GREY LINE

    ax1.axvline(m_IN_max, color="grey", lw=2)
    ax2.axvline(m_IN_max, color="grey", lw=2)
    ax3.axvline(m_IN_max, color="grey", lw=2)
    ax2.annotate("Most massive \nhalo", xy=(m_IN_max, 0.0045), xycoords='data', xytext=(m_IN_max+ 0.1 * 10**14, 0.0045),
                 textcoords='data', horizontalalignment='center',
                 verticalalignment='bottom', rotation="vertical", color="grey", fontsize=14)

    # SHOW LARGEST HALO WITH ARROW

    # y_posarrow_1 = importance[:50][(np.abs(mass_scales-m_IN_max)).argmin()] + \
    #              (0.5* importance[:50][(np.abs(mass_scales-m_IN_max)).argmin()])
    # y_posarrow_2 = importance[50:100][(np.abs(mass_scales-m_IN_max)).argmin()] + \
    #              (0.5* importance[50:100][(np.abs(mass_scales-m_IN_max)).argmin()])
    # y_posarrow_3 = importance[100:150][(np.abs(mass_scales-m_IN_max)).argmin()] + \
    #              (0.5* importance[100:150][(np.abs(mass_scales-m_IN_max)).argmin()])
    #
    # ax1.annotate("", xy=(m_IN_max, y_posarrow_1), xycoords='data',
    #              xytext=(m_IN_max, y_posarrow_1*2.3), textcoords='data',
    #              arrowprops=dict(arrowstyle="simple", facecolor='red', edgecolor="k"), horizontalalignment='center',
    #              verticalalignment='center')
    # ax2.annotate("", xy=(m_IN_max, y_posarrow_2), xycoords='data',
    #              xytext=(m_IN_max, y_posarrow_2*3), textcoords='data',
    #              arrowprops=dict(arrowstyle="simple", facecolor='red', edgecolor="k"), horizontalalignment='center',
    #              verticalalignment='center')
    # ax3.annotate("", xy=(m_IN_max, y_posarrow_3), xycoords='data',
    #              xytext=(m_IN_max, y_posarrow_2*3), textcoords='data',
    #              arrowprops=dict(arrowstyle="simple", facecolor='red', edgecolor="k"), horizontalalignment='center',
    #              verticalalignment='center')

if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    w = window.WindowParameters(initial_parameters=ic)

    ####################### DENSITY #######################

    imp_den = np.load("/Users/lls/Desktop/importances_density_only.npy")
    f_imp_den = np.load("/Users/lls/Documents/CODE/stored_files/all_out/feature_importances/original_feature_set/"
                        "ten_importances_50k_training.npy")
    f_imp_den = np.load("/Users/lls/Documents/CODE/stored_files/shear/importances_10_runs_density_only.npy")
    f_imp_den = np.load("/Users/lls/Documents/CODE/stored_files/shear/importances_10_runs_density_only_less_estimators"
                        ".npy")
    plot_importances(np.mean(f_imp_den, axis=0), w.smoothing_masses, ic, figsize=(10, 5.2),
                      yerr=np.std(f_imp_den, axis=0), ecolor="k", label=None)
    plt.savefig("/Users/lls/Documents/mlhalos_paper/density_importances_no_label.pdf")


    ######################## SHEAR #######################

    # imp_shear_best_fit = np.load("/Users/lls/Documents/CODE/stored_files/shear/"
    #                              "feature_importances_den+den_sub_ell+den_sub_prol.npy")
    # imp_ten_runs = np.load("/Users/lls/Documents/CODE/stored_files/shear/importances_10_runs_den+den_sub_ell"
    #                        "+den_sub_prol.npy")
    #
    # imp_ten_runs = np.load("/Users/lls/Documents/CODE/stored_files/shear/importances_10_runs_den+den_sub_ell"
    #                        "+den_sub_prol_max_feat_auto.npy")
    # plot_density_shear_importance(np.mean(imp_ten_runs, axis=0), w.smoothing_masses, ic, figsize=(10, 5.2),
    #                               yerr=np.std(imp_ten_runs, axis=0), ecolor="k")
    # plt.savefig("/Users/lls/Documents/mlhalos_paper/wide_shear_density_importances.pdf")


    # for i in range(10):
    #     plot_importances(imp_ten_runs[i], w.smoothing_masses, ic, figsize=(10, 5.2), ecolor="k",
    #                      label=["Density", "Ellipticity", "Prolateness"],
    #                      subplots=3)
    #     plt.show()
    #     plt.clf()
    #
    #
    #
    # imp_den_shear = np.load("/Users/lls/Documents/CODE/stored_files/shear/feature_importances_den+den_sub_ell+den_sub_prol.npy")
    # plot_density_shear_importance(imp_den_shear, w.smoothing_masses, figsize=(10, 5.2), ecolor="k")
    # # plt.savefig("/Users/lls/Documents/mlhalos_paper/wide_shear_density_importances.pdf")


