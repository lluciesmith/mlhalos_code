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

path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/reduced_training_set/"
ids_tested = np.load(path + "ids_tested.npy")

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
halo_mass_tested = np.log10(halo_mass[ids_tested])
ids_in_halo_mass = np.where(halo_mass!=0)[0]
n_tot, b_tot = np.histogram(np.log10(halo_mass[halo_mass!=0]), bins=50)

all_ind = []
for i in range(len(b_tot) - 1):
    ind_bin = np.where((halo_mass_tested >= b_tot[i]) & (halo_mass_tested < b_tot[i + 1]))[0]
    if len(ind_bin)!=0:
        if len(ind_bin)<1000:
            num_p = len(ind_bin)
        else:
            num_p = 1000
        ind_ran = np.random.choice(ind_bin, num_p, replace=False)
        all_ind.append(ind_ran)

all_ind = np.concatenate(all_ind)

plt.figure()
plt.hist(np.log10(halo_mass[ids_tested[all_ind]]), bins=30)
plt.xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$")
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/local_inertia/reduced_training_set/subset"
            "/subset_ids_tested.png")

log_true_mass = np.load(path + "true_test_halo_mass.npy")
log_true_mass = np.log10(log_true_mass[all_ind])
bins_plotting = np.linspace(log_true_mass.min(), log_true_mass.max(), 15, endpoint=True)

den_plus_local_inertia = np.load(path + "predicted_halo_mass.npy")
den_plus_inertia_log = np.log10(den_plus_local_inertia[all_ind])

den_predicted = np.load(path + "den_only/predicted_halo_mass.npy")
log_den_predicted = np.log10(den_predicted[all_ind])

path = "/Users/lls/Documents/mlhalos_files/regression/local_inertia/reduced_training_set/subset/"

###### VIOLIN PLOTS #####

plt.figure()
violin_plots(den_plus_inertia_log, log_true_mass, log_den_predicted, log_true_mass,
             bins_plotting, label1="local inertia+density", label2="density only", color1="magenta",
             color2="grey",
             path=path,
             saving_name="density_vs_loc_inertia_plus_den.png")

###### COMPARE PREDICTIONS WITH TRUE MASS #####

x = log_true_mass
y_inden = den_plus_inertia_log
y_den = log_den_predicted

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(11, 5))
fig.subplots_adjust(hspace=0.3, left=0.07, right=0.93, top=0.94)
ax = axs[0]
hb = ax.hexbin(x[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], gridsize=50, cmap='inferno',
               # bins="log",
               # vmin=0, vmax=4.5
               )
ax.plot(x[(x <= 14) & (x > 11)], x[(x <= 14) & (x > 11)], color="k")
ax.set_ylim(11, 14)
ax.set_title("Local Inertia+density")
ax.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", size=17)
ax.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", size=17)
cb = fig.colorbar(hb, ax=ax)
cb.set_label(r'$N$')

ax = axs[1]
hb = ax.hexbin(x[(x <= 14) & (x > 11)], y_den[(x <= 14) & (x > 11)], gridsize=50, cmap='inferno',
               # bins="log",
               # vmin=0, vmax=4.5
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
g = (sns.jointplot(x[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], kind="kde", stat_func=None)) \
    .set_axis_labels("$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", r"$\log (M_\mathrm{predicted}/\mathrm{M}_{"
                                                                    r"\odot})$", size=17)
g.ax_joint.plot([10.5, 14.5], [10.5, 14.5], "k")
plt.savefig(path + "dplot_loc_in_plus_den_vs_true.png")
plt.clf()

plt.figure(figsize=(6.9, 5.2))
g = (sns.jointplot(y_den[(x <= 14) & (x > 11)], y_inden[(x <= 14) & (x > 11)], kind="kde", stat_func=None)) \
    .set_axis_labels("$\log (M_\mathrm{density}/\mathrm{M}_{\odot})$", r"$\log (M_\mathrm{"
                                                                       r"loc.inertia+density}/\mathrm{M}_{"
                                                                       r"\odot})$", size=17)
g.ax_joint.plot([10.5, 14.5], [10.5, 14.5], "k")
plt.savefig(path + "dplot_loc_in_plus_den_vs_den.png")

