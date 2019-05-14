import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from scripts.paper_plots import misclassified_vs_halo_mass as mh
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 18})
import matplotlib.pyplot as plt
from mlhalos import distinct_colours as dc
from mlhalos import parameters
from matplotlib.ticker import MaxNLocator
import math


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def find_nearest(array,value):
    if isinstance(value, list):
        idx = [(np.abs(array-val).argmin()) for val in value]
    else:
        idx = (np.abs(array-value)).argmin()
    return idx


def poisson_propagated_errorbars(number, total_number):
    # err = ((np.sqrt(number) * total_number) + (np.sqrt(total_number * number)))/total_number**2
    err = (number/total_number)*np.sqrt((1/number) + (1/total_number))
    return err


# path = "/Users/lls/Documents/CODE/stored_files/shear/"
path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"
y_pred_den = np.load(path + "density_only/predicted_den.npy")
y_true_den = np.load(path + "density_only/true_den.npy")

# Find FPs and FNs of density run

th = np.linspace(0, 1, 50)[::-1]
ids_tested, halos_testing_particles = mh.get_ids_and_halos_test_set(
    #path="/Users/lls/Documents/CODE/stored_files/classification/"
)

FPs_den_thr = mh.false_positives_ids_index_per_threshold(y_pred_den, y_true_den, th)
FNs_den_thr = mh.false_negatives_ids_index_per_threshold(y_pred_den, y_true_den, th)

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
m_threshold = ic.halo[ic.max_halo_number]['mass'].sum()

thresholds_plot_fp = find_nearest(th, [0.7, 0.6, 0.5, 0.4])
thresholds_plot_fn = find_nearest(th, [0.3, 0.4, 0.5, 0.6])
# thresholds_plot = np.where(th==th_plot)[0]

colors1 = dc.get_distinct(4)
# colors = ['#191966',"#0066cc",'#ff66ff',"#ffb3ff"]
colors = ['#191966',"#0066cc",colors1[1],"#edad5e"]
f, (ax1, ax2) = plt.subplots(2, 1,
                             #figsize=(8, 6),
                             sharex=True)
labels = [0.7, 0.6, 0.5, 0.4]

for i in [3,2,1,0]:
    threshold_fn = thresholds_plot_fn[i]
    col = colors[i]
# for threshold_fn in thresholds_plot:
    # threshold_fn = 35
    # col="k"
    threshold_fp = thresholds_plot_fp[i]
    h_fps = halos_testing_particles[FPs_den_thr[threshold_fp]]
    h_fns = halos_testing_particles[FNs_den_thr[threshold_fn]]

    bins_out = 15
    bins_in = mh.bin_in_equal_number_of_halos(halos_testing_particles[y_true_den == 1], bins_try=8)
    bins_in = np.log10(bins_in)

    n_total_out, bins_total_out = np.histogram(np.log10(halos_testing_particles[(y_true_den == -1) &
                                                                               (halos_testing_particles > 0)]),
                                               bins=bins_out)
    n_total_in, bins_total_in = np.histogram(np.log10(halos_testing_particles[y_true_den == 1]), bins=bins_in)

    FPs_n_den, bins1 = np.histogram(np.log10(h_fps[h_fps > 0]), bins=bins_total_out)
    FNs_n_den, bins2 = np.histogram(np.log10(h_fns), bins=bins_total_in)

    #norm_fp = np.sum(FPs_n_den) * np.diff(bins1)
    #norm_fn = np.sum(FNs_n_den) * np.diff(bins2)

    FPs_fraction = FPs_n_den/ n_total_out
    FNs_fraction = FNs_n_den/ n_total_in

    FNs_yerr = poisson_propagated_errorbars(FNs_n_den, n_total_in)
    FPs_yerr = poisson_propagated_errorbars(FPs_n_den, n_total_out)

    bins1_mid = (bins1[1:] + bins1[:-1]) / 2
    bins2_mid = (bins2[1:] + bins2[:-1]) / 2

    delta_m_fps = np.abs(bins1_mid - np.log10(m_threshold))
    delta_m_fns = np.abs(bins2_mid - np.log10(m_threshold))

    ax1.errorbar(delta_m_fps, FPs_fraction, color=col, lw=1.5, yerr=FPs_yerr,
                 label=r"$P= $" + str(int(labels[i]*100)) + r"$\% $")
    ax2.errorbar(delta_m_fns, FNs_fraction, color=col, lw=1.5, yerr=FNs_yerr)
    # ax1.setp(line, label=r"$\mathrm{P}=$" + str(int(labels[i]*100)) + r"$\% $")
    # ax1.bar(delta_m_fps, FPs_fraction, color=col, label=r"$P= $" + str(int(labels[i]*100)) + r"$\% $")
    # ax2.bar(delta_m_fns, FNs_fraction, color=col)


    ax2.set_xlabel(r"$|\log_{10}(M_{\mathrm{true}}/\mathrm{M_{boundary}})|$")
    ax1.set_ylabel("False Positive Rate", fontsize=17)
    ax2.set_ylabel("False Negative Rate", fontsize=17)
    # ax1.set_ylabel("FPR per mass bin", fontsize=17)
    # ax2.set_ylabel("FNR per mass bin", fontsize=17)
    plt.subplots_adjust(hspace=0)

    # for tick in ax2.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)

    #ax1.set_xscale("log")
    #ax2.set_xscale("log")
    # length_ticks=4
    # ax1.locator_params(axis='y', nticks=length_ticks)
    # ax2.locator_params(axis='y', nticks=length_ticks)
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=length_ticks))
    # ax2.yaxis.set_major_locator(MaxNLocator(nbins=length_ticks, prune='upper'))

length_ticks = 5
ytick1 = ax1.get_yticks()
#tick1_new = np.linspace(ytick1[0], ytick1[-1], length_ticks,endpoint=True)
tick1_new = np.linspace(0, 0.5, length_ticks,endpoint=True)
ax1.set_yticks(tick1_new)

ytick2 = ax2.get_yticks()
tick2_new = np.linspace(ytick2[0], ytick2[-1], length_ticks,endpoint=True)
#tick2_new = np.linspace(0, 1, length_ticks,endpoint=True)
ax2.set_yticks(tick2_new)

ax1.yaxis.set_major_locator(MaxNLocator(nbins=length_ticks))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=length_ticks, prune='upper'))

handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.legend(handles, labels,
           fontsize=14,
           loc="best", frameon=False)

ax1.text(0.15, 0.84, 'Misclassified OUT particles', color="k", multialignment='center', fontsize=16,
        transform=ax1.transAxes)
ax2.text(0.15, 0.84, 'Misclassified IN particles', color="k", multialignment='center', fontsize=16,
        transform=ax2.transAxes)
ax1.set_xlim(0, delta_m_fps.max())
ax2.set_xlim(0, delta_m_fps.max())
plt.savefig("/Users/lls/Documents/mlhalos_paper/misclassified_mass_distance_prob_same_2.pdf")


# NEED ERRORBARS
    #
    # norm_fp = np.sum(FPs_fraction) * np.diff(10**bins1)
    # norm_fn = np.sum(FNs_fraction) * np.diff(10**bins2)
    #
    # plt.plot((bins1[1:] + bins1[:-1]) / 2, FPs_n_den/norm_fp, color=col)
    # plt.plot((bins2[1:] + bins2[:-1]) / 2, FNs_n_den/norm_fn, color=col)


    # try putting them all together

    # miscl = np.concatenate((h_fps, h_fns))
    # # micl_n_den, bins_miscl, p1f = plt.hist(np.log10(miscl[miscl > 0]), bins=np.concatenate((bins_total_out,bins_total_in)),
    # #                                                                                         normed=True)
    # micl_n_den, bins_miscl = np.histogram(np.log10(miscl[miscl > 0]), bins=np.concatenate((bins_total_out, bins_total_in)))
    # norm = np.sum(micl_n_den)*np.diff(bins_miscl)
    # err = np.sqrt(micl_n_den)/norm
    #
    # f, ax = plt.figure(figsize=(8,6))
    # ax.set_color_cycle(colors)
    # plt.errorbar((bins_miscl[1:] + bins_miscl[:-1])/2, micl_n_den/norm, yerr=err)
    # plt.xlabel(r"$\log_{10}(M_{\mathrm{true}}/\mathrm{M}_{\odot})$", fontsize=20)
    # plt.ylabel("PDF misclassified particles", fontsize=20)



