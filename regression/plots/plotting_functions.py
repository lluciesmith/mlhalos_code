import numpy as np
import matplotlib.pyplot as plt
from mlhalos import parameters
from mlhalos import window
from mlhalos import plot
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_predicted_masses_in_each_true_m_bin(bins, mass_predicted_particles, true_mass_particles,
                                            return_stats="median"):
    log_pred_bins = []
    mean_each_bin = []

    for i in range(len(bins) - 1):
        indices_each_bin = np.where((true_mass_particles >= bins[i]) & (true_mass_particles < bins[i + 1]))[0]
        indices_each_bin = indices_each_bin.astype("int")

        predicted_each_bin = mass_predicted_particles[indices_each_bin]
        if return_stats == "median":
            predicted_mean = np.median(predicted_each_bin)
        else:
            predicted_mean = np.mean(predicted_each_bin)

        log_pred_bins.append(list(predicted_each_bin))
        mean_each_bin.append(predicted_mean)

    mean_each_bin = np.array(mean_each_bin)
    return log_pred_bins, mean_each_bin


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def violin_plot_w_percentile(xbins, data_per_bin1, data_per_bin2, color_violin1="green", color_violin2="red", label=[" ", " "]):
    width_xbins = np.diff(xbins)
    xaxis = (xbins[:-1] + xbins[1:])/2
    offset = 0.2 * xaxis / 100

    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    parts = ax2.violinplot(data_per_bin1, positions=xaxis, widths=width_xbins,
                           showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(color_violin1)
        pc.set_edgecolor('black')
        #pc.set_alpha(1)
    d = np.array([np.percentile(bla, [25, 50, 75]) for bla in data_per_bin1])
    quartile1, medians, quartile3 = d[:,0], d[:,1], d[:,2]
    ax2.scatter(xaxis-offset, medians, marker='o', color=color_violin1, s=20, zorder=3, label=label[0])
    ax2.vlines(xaxis-offset, quartile1, quartile3, color=color_violin1, linestyle='-', lw=2)

    parts2 = ax2.violinplot(data_per_bin2, positions=xaxis, widths=width_xbins,
                           showmeans=False, showmedians=False, showextrema=False)

    for pc in parts2['bodies']:
        pc.set_facecolor(color_violin2)
        pc.set_edgecolor('black')
        #pc.set_alpha(1)
    d = np.array([np.percentile(bla, [25, 50, 75]) for bla in data_per_bin2])
    quartile1, medians, quartile3 = d[:,0], d[:,1], d[:,2]
    ax2.scatter(xaxis +offset, medians, marker='x', color=color_violin2, s=20, zorder=3, label=label[1])
    ax2.vlines(xaxis+offset, quartile1, quartile3, color=color_violin2, linestyle='-', lw=2)

    #ax2.vlines(xaxis, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    ax2.plot(xbins, xbins, color="k")
    plt.legend(loc="best")

    ax2.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    ax2.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)


def plot_quantile(xbins, data_per_bin1, data_per_bin2, color1="green", color2="red", label=[" ", " "]):
    xaxis = (xbins[:-1] + xbins[1:])/2
    offset = 0.2 * xaxis / 100

    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    d = np.array([np.percentile(bla, [25, 50, 75]) for bla in data_per_bin1])
    quartile1, medians, quartile3 = d[:,0], d[:,1], d[:,2]
    ax2.scatter(xaxis-offset, medians, marker='o', color=color1, s=20, zorder=3, label=label[0])
    ax2.vlines(xaxis-offset, quartile1, quartile3, color=color1, linestyle='-', lw=2)

    d = np.array([np.percentile(bla, [25, 50, 75]) for bla in data_per_bin2])
    quartile1, medians, quartile3 = d[:,0], d[:,1], d[:,2]
    ax2.scatter(xaxis +offset, medians, marker='x', color=color2, s=20, zorder=3, label=label[1])
    ax2.vlines(xaxis+offset, quartile1, quartile3, color=color2, linestyle='-', lw=2)

    ax2.plot(xbins, xbins, color="k")
    plt.legend(loc="best")

    ax2.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    ax2.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)


def get_violin_from_distributions_per_bin(xbins, distributions_f_per_bisn, mean_per_bins, label_distr=None):
    width_xbins = np.diff(xbins)
    xaxis = (xbins[:-1] + xbins[1:])/2

    fig, axes = plt.subplots(nrows=1, ncols=1)
    color="b"
    vplot = axes.violinplot(distributions_f_per_bisn, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color) for b in vplot['bodies']]

    axes.errorbar(xaxis, mean_per_bins, xerr=width_xbins/2, color="b", fmt="o", label=label_distr)
    #axes.step(xbins[:-1], mean_distr, where="post", color=color, label=label_distr)
    #axes.plot([xbins[-2], xbins[-1]], [mean_distr[-1], mean_distr[-1]], color=color)

    axes.plot(xbins, xbins, color="k")
    axes.set_xlim(xbins.min() - 0.1, xbins.max() + 0.1)
    axes.set_ylim(xbins.min() - 0.1, xbins.max() + 0.1)

    axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)


def get_violin_plot_single_prediction(bins, predicted_mass, true_mass, return_mean=False, label_distr=None):
    pred_per_bin, mean_per_bin = get_predicted_masses_in_each_true_m_bin(bins, predicted_mass, true_mass,
                                                                         return_mean=return_mean)
    get_violin_from_distributions_per_bin(bins, pred_per_bin, mean_per_bin, label_distr=label_distr)


def importances_plot(imp, initial_parameters=None, save=False, yerr=None, label=None, subplots=1, figsize=(7.5, 6)):
    if initial_parameters is None:
        initial_parameters = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE",
                                                                    load_final=True)
    w = window.WindowParameters(initial_parameters=initial_parameters)
    mass = w.smoothing_masses

    plot.plot_importances_vs_mass_scale(imp, mass, save=save, yerr=yerr, label=label, path=".", title=None,
                                        width=0.5, log=False, subplots=subplots, figsize=figsize, frameon=False,
                                        legend_fontsize=None, ecolor="k")


def plot_histogram(predicted, true):
    plt.figure(figsize=(8,6))
    n1, b1, p1 = plt.hist(true, histtype="step",  bins=50, label="true")
    n, b, p = plt.hist(predicted, histtype="step",  bins=b1, label="predicted")
    plt.xlabel(r"$\log (M/\mathrm{M}_{\odot})$", size=17)
    plt.ylabel("N particles", size=17)
    plt.legend(loc=2)


def compare_violin_plots(predicted1, true1, predicted2, true2, bins, label1="den+inertia", label2="density",
                         path=None, saving_name=None, color1="#587058", color2="yellow"):
    pred1, mean1 = get_predicted_masses_in_each_true_m_bin(bins, predicted1, true1,
                                                                        return_mean=False)
    pred2, mean2 = get_predicted_masses_in_each_true_m_bin(bins, predicted2, true2,
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


def compare_2d_histograms(x1, y1, x2, y2, title1="density+shear", title2="density", save_path=None, vmin=None,
                          vmax=None, m_ax=(None, None), gridsize=50, x_min=11, x_max=14, y_min=11, y_max=14):
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(11, 5.5))

    ax = axs[0]
    hb = ax.hexbin(x1[(x1 <= x_max) & (x1 > x_min)], y1[(x1 <= x_max) & (x1 > x_min)], gridsize=gridsize, cmap='inferno',
                   vmin=vmin,
                   vmax=vmax)
    ax.plot(x1[(x1 <= x_max) & (x1 > x_min)], x1[(x1 <= x_max) & (x1 > x_min)], color="k")
    if m_ax == (None, None):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # ax.set_ylim(m_ax)
        ax.set_xlim(m_ax)
    ax.set_title(title1)
    ax.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    ax.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    cb = fig.colorbar(hb, ax=ax)
    cb.remove()

    ax = axs[1]
    hb1 = ax.hexbin(x2[(x2 <= x_max) & (x2 > x_min)], y2[(x2 <= x_max) & (x2 > x_min)], gridsize=gridsize, cmap='inferno',
                    vmin=cb.vmin, vmax=cb.vmax,
                    )
    ax.plot(x2[(x2 <= x_max) & (x2 > x_min)], x2[(x2 <= x_max) & (x2 > x_min)], color="k")
    if m_ax == (None, None):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # ax.set_ylim(m_ax)
        ax.set_xlim(m_ax)
    ax.set_title(title2)
    ax.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
    # ax.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
    # cb1 = fig.colorbar(hb1, ax=ax)
    # cb1.set_label(r'$N$')

    fig.subplots_adjust(right=0.9,
                        wspace=0.01,
                        top=0.94, left=0.1)
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar_ax = fig.add_axes([0.87, 0.1, 0.01, 0.9])
    cb1 = fig.colorbar(hb1, cax=cbar_ax)
    # cb1.set_label(r'$N$')
    if save_path is not None:
        plt.savefig(save_path)

