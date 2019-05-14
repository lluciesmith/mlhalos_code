import numpy as np
import sys
sys.path.append('/Users/lls/Documents/CODE/git/mlhalos_code')
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mlhalos import distinct_colours
from utils import plot
from utils import classification_results as res, classification_results
from mlhalos import machinelearning as ml


def get_particles_in_mass_bin(particles=None, mass_bin="small", high_halo=6, mid_halo=78, initial_parameters=None):
    if initial_parameters is None:
        f, h = res.load_final_snapshot_and_halos()
    else:
        f = initial_parameters.final_snapshot

    if particles is None:
        all_particles = f['iord']
        halos_ids = f['grp']
    else:
        all_particles = particles
        particles = particles.astype("int")
        halos_ids = f[particles]['grp']

    if mass_bin == "small":
        particles_in_and_out_in_halo = all_particles[halos_ids >= mid_halo]
        particles_out_no_halo = all_particles[halos_ids == -1]
        particles = np.concatenate((particles_in_and_out_in_halo, particles_out_no_halo))

    elif mass_bin == "mid":
        particles_in = all_particles[(halos_ids >= high_halo) & (halos_ids < mid_halo)]
        particles_out = np.concatenate((all_particles[halos_ids > 400], all_particles[halos_ids == -1]))
        particles = np.concatenate((particles_in, particles_out))

    elif mass_bin == "high":
        particles_in = all_particles[(halos_ids < high_halo) & (halos_ids > -1)]
        particles_out = np.concatenate((all_particles[halos_ids > 400], all_particles[halos_ids == -1]))
        particles = np.concatenate((particles_in, particles_out))

    else:
        raise (NameError, "Invalid mass bin entered")
    return particles


def get_indices_of_particles_array_in_mass_bin(particles, mass_bin="small"):
    f, h = res.load_final_snapshot_and_halos()

    particles = particles.astype("int")
    halos_ids = f[particles]['grp']

    if mass_bin == "small":
        index = np.where((halos_ids> 77) | (halos_ids== -1))[0]

    elif mass_bin == "mid":
        index = np.where(((halos_ids> 6) & (halos_ids< 78)) | (halos_ids > 400) | (halos_ids == -1))[0]

    elif mass_bin == "high":
        index = np.where((halos_ids < 6) | (halos_ids > 400))[0]

    else:
        raise (NameError, "Invalid mass bin entered")
    return index


def extract_classification_results_subset_particles(classification_results, subset_particle_ids):
    boolean_indices = np.in1d(classification_results[:,0], subset_particle_ids)
    ordered_subset_ids = classification_results[:, 0][boolean_indices]
    true_label_subset_ids = classification_results[:, 1][boolean_indices]
    predicted_probabilities_subset_ids = classification_results[:, 2:4][boolean_indices]
    return ordered_subset_ids, true_label_subset_ids, predicted_probabilities_subset_ids


def get_fpr_tpr_auc_mass_bin(results, mass_bin="small", high_halo=6, mid_halo=78):
    subset = get_particles_in_mass_bin(mass_bin=mass_bin, high_halo=high_halo, mid_halo=mid_halo)
    ordered_ids, true_lab, pred_prob = extract_classification_results_subset_particles(results, subset)
    fpr, tpr, auc, threshold = ml.roc(pred_prob, true_lab)
    return fpr, tpr, auc


def get_EPS_fpr_tpr_auc_mass_bin(results, analytic_probabilities_all, mass_bin="small", high_halo=6, mid_halo=78):
    subset = get_particles_in_mass_bin(mass_bin=mass_bin, high_halo=high_halo, mid_halo=mid_halo)
    boolean_indices = np.in1d(results[:, 0], subset)
    pred = analytic_probabilities_all[boolean_indices]
    true = results[:, 1][boolean_indices]
    fpr_EPS, tpr_EPS, auc_EPS, threshold = ml.roc(pred, true)
    return fpr_EPS[0], tpr_EPS[0]



###################### FUNCTIONS TO HISTOGRAM SUBSET OF IDS AS A FUNCTION OF HALO MASS ######################


def get_mass_each_halo(halos, f=None, h=None):
    if f is None and h is None:
        f,h = classification_results.load_final_snapshot_and_halos()
    mass_halos = np.zeros((len(halos),))
    for i in set(halos):
        pos = np.where(halos == i)[0]
        mass_halos[pos] = h[i]['mass'].sum()
    return mass_halos


def get_halo_mass_each_particle(particles, f=None, h=None):
    if f is None and h is None:
        f,h = classification_results.load_final_snapshot_and_halos()
    if particles.dtype != "int":
        particles = particles.astype("int")
    halos_particles = f[particles]['grp']
    mass_halos = get_mass_each_halo(halos_particles, f=f, h=h)
    return mass_halos


def histogram_halo_mass_particles(mass_halos_particles, xscale="log", number_of_bins=10):
    """

    Args:
        mass_halos_particles:
        xscale (str, np.ndarray, list):
        number_of_bins:

    Returns:

    """
    # mass_halos_particles = get_halo_mass_each_particle(particles)
    if xscale == "log":
        bins = np.power(10, np.linspace(np.log10(mass_halos_particles.min()), np.log10(mass_halos_particles.max()),
                                        number_of_bins))
        plt.xscale("log")

    elif xscale == "uniform":
        bins = number_of_bins

    elif xscale == "equal total particles":
        bins = plot.get_log_spaced_bins_flat_distribution(mass_halos_particles, number_of_bins_init=number_of_bins)
        plt.xscale("log")

    elif xscale == "equal total halos":
        mass_halos_unique = np.unique(mass_halos_particles)
        bins = plot.get_log_spaced_bins_flat_distribution(mass_halos_unique,
                                                          number_of_bins_init=number_of_bins)
        plt.xscale("log")
    elif isinstance(xscale, (np.ndarray, list)):
        bins = xscale
        plt.xscale("log")

    else:
        raise NameError("Choose either uniform or uniform-log scale.")

    n, bins = np.histogram(mass_halos_particles, bins=bins)
    return n, bins


def plot_scatter_and_line(bins, ratio, color=None, marker='o', ls='-', xlabel=None, ylabel=None, label=None,
                          legend=None, legend_loc=None):

    plt.scatter(bins, ratio, color=color, marker=marker)
    l, = plt.plot(bins, ratio, color=color, label=label, ls=ls)

    if xlabel is True:
        plt.xlabel(r"$M_{\mathrm{halo}} [M_{\odot}]$")

    if ylabel is True:
        plt.ylabel(r"$N/N_{\mathrm{all}}$")
    elif isinstance(ylabel, str):
        plt.ylabel(ylabel)
    else:
        pass

    if legend is True:
        if legend_loc is None:
            plt.legend(loc="best")
        else:
            plt.legend(loc=legend_loc)
    return l,


def plot_ratio_halo_mass_subset_vs_all(subset_ids_halo_mass, all_ids_halo_mass, xscale="log", number_of_bins=10,
                                       label=None, color=None, xlabel=None, ylabel=None, legend=None, legend_loc=None,
                                       ls=None, marker=None):

    n_all, bins_all = histogram_halo_mass_particles(all_ids_halo_mass, xscale=xscale, number_of_bins=number_of_bins)
    n_subset, bins_subset = histogram_halo_mass_particles(subset_ids_halo_mass, xscale=bins_all,
                                                          number_of_bins=number_of_bins)

    ratio = np.zeros((len(n_all),))
    ratio[n_all != 0] = n_subset[n_all != 0] / n_all[n_all != 0]
    ratio[n_all == 0] = 0
    mean_bins = np.array([np.mean([bins_all[i], bins_all[i + 1]]) for i in range(len(bins_all)-1)])

    l, = plot_scatter_and_line(mean_bins, ratio, color=color, marker=marker, label=label, ls=ls,
                               xlabel=xlabel, ylabel=ylabel, legend=legend, legend_loc=legend_loc)
    return l,


def plot_fraction_subset_particles_per_halo_mass_bin(TPs_particles, FNs_particles, all_particles, number_halos=False,
                                                     number_particles=False, xscale="equal total particles",
                                                     number_of_bins=25, label_TPs="TPs", label_FNs="FNs", marker='o',
                                                     ls='-', legend=True):
    TPs_halo_mass = get_halo_mass_each_particle(TPs_particles)
    FNs_halo_mass = get_halo_mass_each_particle(FNs_particles)
    all_particles_halo_mass = get_halo_mass_each_particle(all_particles)
    # l, = plot_ratio_halo_mass_subset_vs_all(subset_particles_halo_mass, all_particles_halo_mass, xscale=xscale,
    #                                    number_of_bins=number_of_bins, label=label, color=color, xlabel=xlabel,
    #                                    ylabel=ylabel, legend=legend,
    #                                    legend_loc=legend_loc, marker=marker, ls=ls)
    TPs_FNs_binning(all_particles_halo_mass, TPs_halo_mass, FNs_halo_mass, number_halos=number_halos,
                    number_particles=number_particles, xscale=xscale, number_of_bins=number_of_bins,
                    label_TPs=label_TPs, label_FNs=label_FNs, marker=marker, ls=ls, legend=legend)


def plot_TPS_FPs_particles_per_halo_mass_bin(all_particles, TPs, FNs, number_halos=False, number_particles=False,
                                             xscale="equal total particles", label_TPs="TPs", label_FNs="FNs",
                                             number_of_bins=25, marker='o', ls='-', legend=True):
    TPs_halo_mass = get_halo_mass_each_particle(TPs)
    FNs_halo_mass = get_halo_mass_each_particle(FNs)
    all_particles_halo_mass = get_halo_mass_each_particle(all_particles)
    TPs_FNs_binning(all_particles_halo_mass, TPs_halo_mass, FNs_halo_mass, number_halos=number_halos,
                    number_particles=number_particles, xscale=xscale, legend=legend,
                    number_of_bins=number_of_bins, label_FNs=label_FNs, label_TPs=label_TPs, marker=marker, ls=ls)


def plot_ratio_ML_EPS_subset_particles_per_halo_mass_bin(subset_particles_ML, subset_particles_EPS, all_particles,
                                                         xscale="log", number_of_bins=10,
                                                         label=None, color=None, xlabel=None, ylabel=None, legend=None,
                                                         legend_loc=None, marker='o', ls='-'):

    subset_particles_halo_mass_ML = get_halo_mass_each_particle(subset_particles_ML)
    subset_particles_halo_mass_EPS = get_halo_mass_each_particle(subset_particles_EPS)

    all_particles_halo_mass = get_halo_mass_each_particle(all_particles)
    bins_all = plot.get_log_spaced_bins_flat_distribution(all_particles_halo_mass, number_of_bins_init=number_of_bins)
    if xscale == "log":
        plt.xscale("log")

    n_ML, bins_ML = histogram_halo_mass_particles(subset_particles_halo_mass_ML, xscale=bins_all,
                                                  number_of_bins=number_of_bins)
    n_EPS, bins_EPS = histogram_halo_mass_particles(subset_particles_halo_mass_EPS, xscale=bins_all,
                                                  number_of_bins=number_of_bins)

    ratio = n_ML / n_EPS
    mean_bins = np.array([np.mean([bins_all[i], bins_all[i + 1]]) for i in range(len(ratio))])

    l, = plot_scatter_and_line(mean_bins, ratio, color=color, marker=marker, label=label, ls=ls,
                               xlabel=xlabel, ylabel=ylabel, legend=legend, legend_loc=legend_loc)
    return l,


def TPs_FNs_binning(all_particles_halo_mass, TPs_mass, FNs_mass, number_halos=False, number_particles=False,
                    xscale="equal total particles", number_of_bins=25, label_TPs="TPs", label_FNs="FNs", marker='o',
                    ls='-', legend=True):

    n_all, bins_all = histogram_halo_mass_particles(all_particles_halo_mass, xscale=xscale,
                                                    number_of_bins=number_of_bins)
    n_TP, bins_TPs = histogram_halo_mass_particles(TPs_mass, xscale=bins_all)
    n_FN, b_FN = histogram_halo_mass_particles(FNs_mass, xscale=bins_all)
    mass_halos = np.unique(all_particles_halo_mass)
    n_halos, bins_halos = histogram_halo_mass_particles(mass_halos, xscale=bins_all)

    n_all_plot = n_all[n_all != 0]
    n_FN_plot = n_FN[n_all != 0]
    n_TP_plot = n_TP[n_all != 0]
    n_halos_plot = n_halos[n_all != 0]

    mean_bins = np.array([np.mean([bins_all[i], bins_all[i + 1]]) for i in range(len(bins_all) - 1)])
    mean_bins_plot = mean_bins[n_all != 0]

    r_FN = n_FN_plot / n_all_plot
    r_TP = n_TP_plot / n_all_plot

    err_FN = (n_FN_plot / n_all_plot) * np.sqrt((1 / n_FN_plot) + (1 / n_all_plot))
    print("The false negatives errorbars are " + str(err_FN))
    err_TP = (n_TP_plot / n_all_plot) * np.sqrt((1 / n_TP_plot) + (1 / n_all_plot))
    print("The true positives errorbars are " + str(err_TP))

    if number_halos is True and number_particles is True:
        f = plt.figure(figsize=(9, 6))
        plt.subplots_adjust(hspace=0.001)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
        colors = distinct_colours.get_distinct(4)

        ax1 = plt.subplot(gs[0])
        plt.xscale("log")
        ax1.errorbar(mean_bins_plot, r_FN, xerr=err_FN, label="FNs", color=colors[0])
        ax1.scatter(mean_bins_plot, r_FN, color=colors[0])
        ax1.errorbar(mean_bins_plot, r_TP, xerr=err_TP, label="TPs", color=colors[1])
        ax1.scatter(mean_bins_plot, r_TP, color=colors[1])
        ax1.set_ylabel(r"$N/ N_{\mathrm{all}}$")
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend(loc="best")

        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.scatter(mean_bins_plot, n_halos_plot, label="halos", color='b')
        ax2.plot(mean_bins_plot, n_halos_plot, color='b')
        ax2.set_ylabel(r"$N_{\mathrm{halos}}$")
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.scatter(mean_bins_plot, n_all, label="particles", color='b')
        ax3.plot(mean_bins_plot, n_all, color='b')
        ax3.set_yscale("log")
        ax3.set_ylabel(r"$N_{\mathrm{particles}}$")

        plt.xlabel(r"$M_{\mathrm{halo}} [M_{\odot}]$")
        # nbins = len(ax1.get_xticklabels())
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
        ax3.yaxis.set_major_locator(MaxNLocator(prune='upper'))

    else:
        colors = distinct_colours.get_distinct(2)
        plt.xscale("log")
        plt.errorbar(mean_bins_plot, r_FN, xerr=err_FN, label=label_FNs, color=colors[0], ls=ls)
        plt.scatter(mean_bins_plot, r_FN, color=colors[0], marker=marker, )
        plt.errorbar(mean_bins_plot, r_TP, xerr=err_TP, label=label_TPs, color=colors[1], ls=ls)
        plt.scatter(mean_bins_plot, r_TP, color=colors[1], marker=marker)
        plt.ylabel(r"$N/ N_{\mathrm{all}}$")
        #plt.ylim(0, 1)
        if legend is True:
            plt.legend(loc="best")
        plt.xlabel(r"$M_{\mathrm{halo}} [M_{\odot}]$")


def plot_number_category_as_mass_function(all_particles_halo_mass, category_mass, xscale="equal total particles",
                                          number_of_bins=25, label="FNs", marker='o', ls='-', legend=True, color='b'):
    n_all, bins_all = histogram_halo_mass_particles(all_particles_halo_mass, xscale=xscale,
                                                    number_of_bins=number_of_bins)
    n_category, bins_category = histogram_halo_mass_particles(category_mass, xscale=bins_all)

    mass_halos = np.unique(all_particles_halo_mass)
    n_halos, bins_halos = histogram_halo_mass_particles(mass_halos, xscale=bins_all)


    n_all_plot = n_all[n_all != 0]
    n_category_plot = n_category[n_all != 0]
    n_halos_plot = n_halos[n_all != 0]

    mean_bins = np.array([np.mean([bins_all[i], bins_all[i + 1]]) for i in range(len(bins_all) - 1)])
    mean_bins_plot = mean_bins[n_all != 0]

    r_cat = n_category_plot / n_all_plot
    err_cat = (n_category_plot / n_all_plot) * np.sqrt((1 / n_category_plot) + (1 / n_all_plot))

    plt.xscale("log")

    plt.errorbar(mean_bins_plot, r_cat, xerr=err_cat, label=label, color=color, ls=ls)
    plt.scatter(mean_bins_plot, r_cat, color=color, marker=marker, )
    plt.ylabel(r"$N_{\mathrm{FNs}}/ N_{\mathrm{all}}$")
    if legend is True:
        plt.legend(loc="best")
    plt.xlabel(r"$M_{\mathrm{halo}} [M_{\odot}]$")


