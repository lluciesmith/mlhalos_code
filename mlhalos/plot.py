"""
:mod:`plot`

Contains all functions relative to making plots.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import collections

from . import window
from . import parameters
from . import distinct_colours
from matplotlib.ticker import MaxNLocator, LogLocator


def plot_trajectories(mass_spheres, density_contrast_in, density_contrast_out, num_particles=None, parameters=None,
                      max_number_trajectories=20, threshold=True, convergence=True,
                      mean_convergence=1.004, num_trajectory="multiple",
                      in_label= "IN class", out_label="OUT class", ls_in="-", ls_out="-", alpha=1, figsize=None):
    """
    Plots trajectories of particles.

    Args:
         parameters (class): Instance of :class:`InitialConditionsParameters`. Code looks if
             parameters.num_particles was specified.
         mass_spheres (array): 1D array of smoothing top-hat filter masses.
         density_contrast_in (ndarray): array of densities at each top-hat filter smoothing scale for
             each in particle.
         density_contrast_out (ndarray): array of densities at each top-hat filter smoothing scale for
             each out particle.
         num_particles (int, None): Specify number of random particles to visualise trajectories if not specified in
             parameters.
         max_number_trajectories (int): Set a maximum number of trajectories per plot. Default is 20 in and 20 out.

    Returns:
        Trajectories plot.
    """

    if num_trajectory == "single":
        figure, axis = plt.subplots(figsize=figsize)
        col = ["#cc6600", "#004466"]
        axis.plot(mass_spheres, density_contrast_in, color=col[0])

        if threshold is True:
            density_threshold = axis.plot([int(min(mass_spheres)), int(max(mass_spheres))], [1.0169, 1.0169],
                                          linestyle='dashed', color='k',
                                          # label='threshold'
                                          )

        if convergence is True:
            density_convergence = axis.plot([int(min(mass_spheres)), int(max(mass_spheres))], [1, 1], 'k-',
                                            # label='convergence'
                                            )

        plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        axis.set_xlabel(r"$M_{\mathrm{smoothing}}/\mathrm{M}_{\odot}$")
        axis.set_ylabel(r"$\delta + 1$")
        handles, labels = axis.get_legend_handles_labels()
        axis.set_xlim(int(min(mass_spheres)), int(max(mass_spheres)))
        axis.legend(handles, labels)
        axis.set_xscale('log')

    else:
        if num_particles is not None:

            # Define a function that returns a random subset of number_particles densities.
            def random_subset(density, parameters=parameters, number_particles=num_particles):
                if number_particles is None:
                    number_particles = parameters.num_particles

                if number_particles > 20:
                    raise ValueError('Too many trajectories to evaluate. Choose a smaller number of particles')
                elif number_particles == 0:
                    raise ValueError('You have chosen 0 particles. Please enter number of particles')

                density_subset = density[np.random.choice(range(len(density)), num_particles)]
                return density_subset

            # Take number_particles random subset of in and out trajectories.
            density_contrast_in = random_subset(density_contrast_in, parameters=parameters,
                                                number_particles=num_particles)
            density_contrast_out = random_subset(density_contrast_out, parameters=parameters,
                                                 number_particles=num_particles)

        assert (len(density_contrast_in) <= max_number_trajectories) and \
               (len(density_contrast_out) <= max_number_trajectories), \
            "Please take a subset of particles within the maximum number of trajectories, or raise the " \
            "value of maximum number of trajectories."

        figure, axis = plt.subplots(figsize=figsize)
        col = distinct_colours.get_distinct(2)
        col = ["#cc6600","#004466"]
        in_trajectories = [axis.plot(mass_spheres, density_contrast_in[i], color=col[0], ls=ls_in,
                                     lw=1.5, alpha=alpha, label=in_label if i==0 else "")
                           for i in range(len( density_contrast_in))]
        out_trajectories = [axis.plot(mass_spheres, density_contrast_out[i], color=col[1], ls=ls_out, lw=1.5,
                                      alpha=alpha, label=out_label if i == 0 else "")
                            for i in range(len(density_contrast_out))]

        if threshold is True:
            density_threshold = axis.plot([int(min(mass_spheres)), int(max(mass_spheres))], [1.0169, 1.0169],
                                          linestyle='dashed', color='k',
                                          # label='threshold'
                                          )

        if convergence is True:
            density_convergence = axis.plot([int(min(mass_spheres)), int(max(mass_spheres))],
                                            [mean_convergence, mean_convergence], 'k-')

        plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        axis.set_xlabel(r"$M_\mathrm{smoothing}/\mathrm{M}_{\odot}$")
        axis.set_ylabel(r"$\delta + 1$")
        handles, labels = axis.get_legend_handles_labels()
        # axis.set_xlim(int(min(mass_spheres)), int(max(mass_spheres)))
        axis.set_xlim(mass_spheres.min(), mass_spheres.max())
        axis.legend(handles, labels, frameon=False, fontsize=16)
        axis.set_xscale('log')


def plot_histogram(delta_in, delta_out):
    """Plot histograms of densities for in and out particles"""
    figure, axis = plt.subplots()

    axis.hist(delta_in, label='IN halo', histtype='step', normed=True)
    axis.hist(delta_out, label='OUT halo', histtype='step', normed=True)

    axis.legend(loc='best')
    axis.set_xlabel('density')
    axis.set_ylabel('number particles')


def plot_tsne_features(features_in, features_out):
    """t-SNE plot of two-dimensional features for in and out particles"""
    figure, axis = plt.subplots(figsize=(12.35, 7.8))

    axis.scatter(features_in[:, 0], features_in[:, 1], color='b', cmap=plt.get_cmap('Spectral'), label='In halos')
    axis.scatter(features_out[:, 0], features_out[:, 1], color='g', cmap=plt.cm.get_cmap('Spectral'), label='Out halos')

    axis.legend(loc=3)
    axis.set_xlabel("Feature 1", fontsize=25)
    axis.set_ylabel("Feature 2", fontsize=25)
    plt.rc('xtick', labelsize=23)
    plt.rc('ytick', labelsize=23)
    return figure


def roc_plot(fpr, tpr, auc, labels=[" "],
             figsize=(8, 6),
             add_EPS=False, fpr_EPS=None, tpr_EPS=None, label_EPS="EPS",
             add_ellipsoidal=False, fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_ellipsoidal="ST ellipsoidal",
             frameon=False, fontsize_labels=20, cols=None):
    """Plot a ROC curve given the false positive rate(fpr), true positive rate(tpr) and Area Under Curve (auc)."""

    if figsize is not None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure, ax = plt.subplots()

    if cols is None:
        if len(fpr.shape) > 1:
            cols = distinct_colours.get_distinct(fpr.shape[1])
        else:
            cols = distinct_colours.get_distinct(1)[0]
    ax.set_color_cycle(cols)

    ax.plot(fpr, tpr, lw=1.5)
    ax.set_xlabel('False Positive Rate', fontsize=fontsize_labels)
    ax.set_ylabel('True Positive Rate', fontsize=fontsize_labels)


    # Robust against the possibility of AUC being a single number instead of list
    if not isinstance(auc, (collections.Sequence, list, np.ndarray)):
        auc = [auc]

    if len(labels) > 0:
        labs = []
        for i in range(len(labels)):
            labs.append(labels[i] + " (AUC = " + ' %.3f' % (auc[i]) + ")")
    else:
        labs = np.array(range(len(ax.lines)), dtype='str')
        for i in range(len(labs)):
            labs[i] = (labs[i] + " (AUC = " + ' %.3f' % (auc[i]) + ")")

    if add_EPS is True:
        plt.scatter(fpr_EPS, tpr_EPS, color="k", s=30)
        if label_EPS is not None:
            labs.append(label_EPS)

    if add_ellipsoidal is True:
        if len([fpr_ellipsoidal]) > 1:
            plt.scatter(fpr_ellipsoidal[0], tpr_ellipsoidal[0], color="k", marker="^", s=30)
            plt.scatter(fpr_ellipsoidal[1], tpr_ellipsoidal[1], color="r", marker="^", s=30)
            labs.append("ST ellipsoidal, a=0.75")
            labs.append("ST ellipsoidal, a=0.707")
        else:
            plt.scatter(fpr_ellipsoidal, tpr_ellipsoidal, color="k", marker="^", s=30)
            if label_ellipsoidal is not None:
                labs.append(label_ellipsoidal)

    ax.legend(labs, loc=4,
               # fontsize=18,
               bbox_to_anchor=(0.95, 0.05), frameon=frameon)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)

    return figure


def plot_feature_importance_ranking(importance, indices=None, color="r", label="", legend=False, xlabel=None, yerr=None,
                                    title=False, fontsize=15):
    """ Plot the feature importances of the forest """

    fig, ax = plt.subplots(figsize=(19, 6))

    if indices is None:
        indices = np.argsort(importance)[::-1]

    if title is True:
        ax.set_title("Feature importance")

    if yerr is not None:
        ax.bar(range(len(importance)), importance[indices], yerr=yerr[indices], color=color, align="center",
               label=label)
    else:
        ax.bar(range(len(importance)), importance[indices], color=color, align="center", label=label)
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels(indices, fontsize=fontsize)
    ax.set_xlim([-1, len(importance)])

    if legend is True:
        plt.legend(loc="best")

    if xlabel is None:
        ax.set_xlabel("Features", labelpad=15)
    else:
        ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel("Importance", labelpad=19)

    return fig


def plot_EPS_feature_imp_as_func_mass(importances, initial_parameters,
                                      num_filtering_scales=50, xlim=None):
    w1 = window.WindowParameters(initial_parameters=initial_parameters, num_filtering_scales=num_filtering_scales)
    m1 = w1.smoothing_masses

    fig = plt.figure(figsize=(13,5))

    plt.scatter(m1, importances)

    if xlim is not None:
        plt.xlim(0, xlim)

    plt.xlabel('top-hat filter mass')
    plt.ylabel('importance')
    plt.title("feature importance" )

    return fig


def plot_importances_vs_scale_number(importance, subplots=0, save=False, color="r", label="density", path=".",
                                     title=None, figsize=(16, 10)):
    if subplots == 3:

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=figsize, sharex=True)
        scales = 50
        color = distinct_colours.get_distinct(3)

        ax1.bar(range(scales), importance[:50], color=color[0], align="center", label=label[0])
        ax2.bar(range(scales), importance[50:100], color=color[1], align="center", label=label[1])
        ax3.bar(range(scales), importance[100:150], color=color[2], align="center", label=label[2])

        fig.subplots_adjust(hspace=0)
        fig.text(0.06, 0.5, 'Importance', fontsize=19, rotation='vertical')
        ax3.set_xlabel("Features", labelpad=15)

        ax3.legend(loc="best", fontsize=18)
        ax2.legend(loc="best", fontsize=18)
        ax1.legend(loc="best", fontsize=18)

        ax3.set_xlim([-1, 51])
        ax3.set_xticks(range(50))
        ax3.set_xticklabels(range(50), fontsize=15)

        #ax1.set_title(str(label[0]) + " + " + str(label[1]) + " + " + str(label[2])+ " classification run")
        if title is not None:
            ax1.set_title(title)

        x1,x2,y1,y21 = ax1.axis()
        x1, x2, y1, y22 = ax2.axis()
        x1, x2, y1, y23 = ax3.axis()
        y_max = np.max([y21, y22, y23])
        ax1.set_ylim([0, y_max])
        ax2.set_ylim([0, y_max])
        ax3.set_ylim([0, y_max])

        if save is True:
            plt.savefig(path + "/f_imp.pdf")

    elif subplots == 2:

        fig, (ax1, ax2) = plt.subplots(2, figsize=figsize, sharex=True)
        scales = 50
        color=distinct_colours.get_distinct(2)

        ax1.bar(range(scales), importance[:50], color=color[0], align="center", label=label[0])
        ax2.bar(range(scales), importance[50:100], color=color[1], align="center", label=label[1])

        fig.subplots_adjust(hspace=0)
        fig.text(0.06, 0.5, 'Importance', fontsize=19, rotation='vertical')
        ax2.set_xlabel("Features", labelpad=15)

        ax2.legend(loc="best", fontsize=18)
        ax1.legend(loc="best", fontsize=18)

        ax2.set_xlim([-1, 51])
        ax2.set_xticks(range(50))
        ax2.set_xticklabels(range(50), fontsize=15)

        x1,x2,y1,y21 = ax1.axis()
        x1, x2, y1, y22 = ax2.axis()
        y_max = np.max([y21, y22])
        ax1.set_ylim([0, y_max])
        ax2.set_ylim([0, y_max])

        #ax1.set_title(str(label[0]) + " + " + str(label[1]) + " classification run")

        if save is True:
            plt.savefig(path + "/f_imp.pdf")

    else:
        fig, ax = plt.subplots(figsize=figsize, sharex=True)

        scales = len(importance)
        ax.bar(range(scales), importance, color=color, align="center", label=label)

        ax.set_ylabel("Importance")
        ax.set_xlabel("Features")

        ax.legend(loc="best", fontsize=18)

        ax.set_xlim([-1, scales + 1])
        ax.set_xticks(range(scales))
        ax.set_xticklabels(range(scales), fontsize=15)
        if save is True:
            plt.savefig(path + "/f_imp.pdf")


def plot_importances_vs_mass_scale(importance, mass, save=False, yerr=None, label="Density",
                                   path=".", title=None, width=0.5, log=False, subplots=3, figsize=(6.9, 5.2),
                                   frameon=False, legend_fontsize=None, ecolor="k"):
    color = distinct_colours.get_distinct(3)

    if subplots == 3:

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=figsize, sharex=True)
        c2 = distinct_colours.get_distinct(4)[3]

        width = np.append(np.diff(mass), np.diff(mass)[-1])
        if yerr is not None:
            ax1.bar(mass, importance[:len(mass)], color=color[0], width=width*2/3, align="center", label=label[0],
                    log=log,
                    yerr=yerr[:len(mass)], ecolor=ecolor, edgecolor=ecolor)
            ax2.bar(mass, importance[len(mass):len(mass)*2], color="#ffbf80", width=width*2/3, align="center",
                    label=label[1], log=log,
                    yerr=yerr[len(mass):len(mass)*2], ecolor=ecolor, edgecolor=ecolor)
            ax3.bar(mass, importance[len(mass)*2:len(mass)*3], color="#669999", width=width*2/3, align="center",
                    label=label[2], log=log,
                    yerr=yerr[len(mass)*2:len(mass)*3], ecolor=ecolor, edgecolor=ecolor)
        else:
            ax1.bar(mass, importance[:50], color=color[0], width=width*2/3, align="center", label=label[0], log=log,
                    ecolor=ecolor)
            ax2.bar(mass, importance[50:100], color="#ffbf80", width=width*2/3,
                    align="center", label=label[1],
                    log=log, ecolor=ecolor)
            ax3.bar(mass, importance[100:150], color="#669999", width=width*2/3, align="center", label=label[2],
                    log=log, ecolor=ecolor)

        fig.subplots_adjust(hspace=0)
        # fig.text(0.05, 0.5, 'Importance', va='center', rotation='vertical', transform=ax2.transAxes)
        ax2.set_ylabel("Importance", fontsize=17)
        ax3.set_xlabel(r"$M_{\mathrm{smoothing}} / \mathrm{M}_{\odot}$",
                      #labelpad=15
                      )

        ax3.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)
        ax2.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)
        ax1.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)

        # ax3.set_xlim([mass.min() - width[0], mass.max() + width[-1]])
        # ax3.set_xticks(mass)
        # ax3.set_xticklabels(mass)
        #
        # ax1.set_xticklabels([])
        # ax2.set_xticklabels([])

        plt.xscale("log")
        #plt.subplots_adjust(bottom=0.15)
        # ax1.xaxis.set_major_locator(LogLocator())
        # ax1.xaxis.set_minor_locator(LogLocator())
        # ax2.xaxis.set_major_locator(LogLocator())
        # ax2.xaxis.set_minor_locator(LogLocator())
        # ax3.xaxis.set_major_locator(LogLocator())
        # ax3.xaxis.set_minor_locator(LogLocator())

        #ax1.set_title(str(label[0]) + " + " + str(label[1]) + " + " + str(label[2])+ " classification run")
        if title is not None:
            ax1.set_title(title)

        # y1 = ax1.get_yticks()
        # y2 = ax2.get_yticks()
        # y3 = ax3.get_yticks()
        # y_max = np.max([y1.max(), y2.max(), y3.max()])
        y_max = importance.max()
        ax1.set_ylim(-0.02, y_max+0.05)
        ax2.set_ylim(ax1.get_ylim())
        ax3.set_ylim(ax1.get_ylim())
        #
        length_ticks = 4
        yticks_change = np.linspace(0, y_max, num=length_ticks, endpoint=True)
        ax1.set_yticks(yticks_change)
        ax2.set_yticks(yticks_change)
        ax3.set_yticks(yticks_change)

        nbins = length_ticks  # added
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))

        if save is True:
            plt.savefig(path + "/f_imp.pdf")

        return fig, ax1, ax2, ax3

    elif subplots == 4:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=figsize, sharex=True)
        c2 = distinct_colours.get_distinct(4)[3]

        width = np.append(np.diff(mass), np.diff(mass)[-1])
        if yerr is not None:
            ax1.bar(mass, importance[:50], color=color[0], width=width*2/3, align="center", label=label[0], log=log,
                    yerr=yerr[:50], ecolor=ecolor)
            ax2.bar(mass, importance[50:100], color="#ffbf80", width=width*2/3, align="center", label=label[1], log=log,
                    yerr=yerr[50:100], ecolor=ecolor)
            ax3.bar(mass, importance[100:150], color="#669999", width=width*2/3, align="center", label=label[2], log=log,
                    yerr=yerr[100:150], ecolor=ecolor)
            ax4.bar(mass, importance[150:200], color=c2, width=width*2/3, align="center", label=label[3], log=log,
                    yerr=yerr[150:200], ecolor=ecolor)
        else:

            ax1.bar(mass, importance[:50], color=color[0], width=width*2/3, align="center", label=label[0], log=log,
                    ecolor=ecolor)
            ax2.bar(mass, importance[50:100], color="#ffbf80", width=width*2/3,
                    align="center", label=label[1],
                    log=log, ecolor=ecolor)
            ax3.bar(mass, importance[100:150], color="#669999", width=width*2/3, align="center", label=label[2],
                    log=log, ecolor=ecolor)
            ax4.bar(mass, importance[150:200], color=c2, width=width*2/3, align="center", label=label[3], log=log,
                    ecolor=ecolor)

        fig.subplots_adjust(hspace=0)
        # fig.text(0.05, 0.5, 'Importance', va='center', rotation='vertical', transform=ax2.transAxes)
        ax2.set_ylabel("Importance", fontsize=17)
        ax4.set_xlabel(r"$M_{\mathrm{smoothing}} / \mathrm{M}_{\odot}$",
                      #labelpad=15
                      )
        ax4.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)
        ax3.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)
        ax2.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)
        ax1.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)

        ax4.set_xlim([mass.min() - width[0], mass.max() + width[-1]])
        ax4.set_xticks(mass)
        ax4.set_xticklabels(mass,
                            #fontsize=15
                            )
        plt.xscale("log")
        ax1.xaxis.set_major_locator(LogLocator())
        ax1.xaxis.set_minor_locator(LogLocator())
        ax2.xaxis.set_major_locator(LogLocator())
        ax2.xaxis.set_minor_locator(LogLocator())
        ax3.xaxis.set_major_locator(LogLocator())
        ax3.xaxis.set_minor_locator(LogLocator())
        ax4.xaxis.set_major_locator(LogLocator())
        ax4.xaxis.set_minor_locator(LogLocator())

        #ax1.set_title(str(label[0]) + " + " + str(label[1]) + " + " + str(label[2])+ " classification run")
        if title is not None:
            ax1.set_title(title)

        # y1 = ax1.get_yticks()
        # y2 = ax2.get_yticks()
        # y3 = ax3.get_yticks()
        # y_max = np.max([y1.max(), y2.max(), y3.max()])
        y_max = importance.max()
        # ax1.set_ylim(0, y_max)

        length_ticks = 4
        yticks_change = np.linspace(0, y_max, num=length_ticks, endpoint=True)
        ax1.set_yticks(yticks_change)
        yticks_change = np.linspace(0, y_max, num=length_ticks, endpoint=True)
        ax2.set_yticks(yticks_change)
        ax3.set_yticks(yticks_change)
        ax4.set_yticks(yticks_change)

        nbins = length_ticks  # added
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        ax4.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        # ax2.set_ylim(0, 0.04)
        # ax3.set_ylim(0, 0.04)

        if save is True:
            plt.savefig(path + "/f_imp.pdf")

        return fig, ax1, ax2, ax3, ax4

    elif subplots == 2:
        fig, (ax1, ax2) = plt.subplots(subplots, figsize=figsize, sharex=True)
        c2 = distinct_colours.get_distinct(4)[3]

        width = np.append(np.diff(mass), np.diff(mass)[-1])
        if yerr is not None:
            ax1.bar(mass, importance[:50], color=color[0], width=width*2/3, align="center", label=label[0], log=log,
                    yerr=yerr[:50], ecolor=ecolor)
            ax2.bar(mass, importance[50:100], color=c2, width=width*2/3, align="center", label=label[1], log=log,
                    yerr=yerr[50:100], ecolor=ecolor)
        else:

            ax1.bar(mass, importance[:50], color=color[0], width=width*2/3, align="center", label=label[0], log=log,
                    ecolor=ecolor)
            ax2.bar(mass, importance[50:100], color=c2, width=width*2/3,
                    align="center", label=label[1],
                    log=log, ecolor=ecolor)

        fig.subplots_adjust(hspace=0)
        # fig.text(0.05, 0.5, 'Importance', va='center', rotation='vertical', transform=ax2.transAxes)
        ax2.set_ylabel("Importance", fontsize=17)
        ax2.set_xlabel(r"$M_{\mathrm{smoothing}} / \mathrm{M}_{\odot}$",
                      #labelpad=15
                      )
        ax2.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)
        ax1.legend(loc=(0.05, 0.6),
                   #fontsize=18,
                   frameon=frameon)

        ax2.set_xlim([mass.min() - width[0], mass.max() + width[-1]])
        ax2.set_xticks(mass)
        ax2.set_xticklabels(mass,
                            #fontsize=15
                            )
        plt.xscale("log")
        # ax1.xaxis.set_major_locator(LogLocator())
        # ax1.xaxis.set_minor_locator(LogLocator())
        # ax2.xaxis.set_major_locator(LogLocator())
        # ax2.xaxis.set_minor_locator(LogLocator())

        #ax1.set_title(str(label[0]) + " + " + str(label[1]) + " + " + str(label[2])+ " classification run")
        if title is not None:
            ax1.set_title(title)

        # y1 = ax1.get_yticks()
        # y2 = ax2.get_yticks()
        # y3 = ax3.get_yticks()
        # y_max = np.max([y1.max(), y2.max(), y3.max()])
        y_max = importance.max()
        # ax1.set_ylim(0, y_max)

        length_ticks = 4
        yticks_change = np.linspace(0, y_max, num=length_ticks, endpoint=True)
        ax1.set_yticks(yticks_change)
        #yticks_change = np.linspace(0, 0.025, num=length_ticks, endpoint=True)
        ax2.set_yticks(yticks_change)

        nbins = length_ticks  # added
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))


        if save is True:
            plt.savefig(path + "/f_imp.pdf")

        return fig, ax1, ax2
    else:
        fig, ax = plt.subplots(figsize=figsize, sharex=True)

        # width = np.append(np.diff(np.log10(mass)), np.diff(np.log10(mass))[-1])
        width = np.append(np.diff(mass), np.diff(mass)[-1])
        ax.bar(mass, importance, color=color[0], width=width*2/3, align="center", label=label, log=log,
               yerr=yerr, ecolor=ecolor, edgecolor=ecolor)
        ax.set_xscale("log")
        #ax.set_xticklabels(mass)

        ax.set_ylabel("Importance", fontsize=17)
        ax.set_xlabel(r"$M_{\mathrm{smoothing}} / \mathrm{M}_{\odot} $",
                      #labelpad=15
                      )

        # ax.legend(loc=(0.05, 0.75),
        #           #fontsize=18,
        #           frameon=frameon)

        ax.legend(loc="best",
                  #fontsize=18,
                  frameon=frameon)

        ax.set_xlim([mass.min() - width[0], mass.max() + width[-1]])
        if title is not None:
            ax.set_title(title)
        #ax.set_xticks(mass)
        #ax.set_xticklabels(mass, fontsize=15)

        if save is True:
            plt.savefig(path + "/f_imp.pdf")



# figure, axis = plt.subplots()
# col = "#cc6600"
# d_in = den_f[int(in_traj), :-1]
# axis.plot(m, d_in, color=col, lw=1.5, label="IN class")
# axis.plot([int(min(mass_spheres)), int(max(mass_spheres))], [1, 1], 'k-')
# plt.rc('text', usetex=True)
# # plt.rc('font', family='serif')
# axis.set_xlabel(r"$M_\mathrm{smoothing}/\mathrm{M}_{\odot}$")
# axis.set_ylabel(r"$\delta + 1$")
# handles, labels = axis.get_legend_handles_labels()
# axis.set_xlim(int(min(mass_spheres)), int(max(mass_spheres)))
# axis.set_ylim(0.98, 1.10)
# axis.legend(handles, labels, frameon=False)
# axis.set_xscale('log')



