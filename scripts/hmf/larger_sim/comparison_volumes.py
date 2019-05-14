import numpy as np
import matplotlib.pyplot as plt
from scripts.hmf.larger_sim import hmf_analysis as ha
from mlhalos import parameters
from mlhalos import distinct_colours
import sys


if __name__ == "__main__":
    kernel = sys.argv[1]
    volume = sys.argv[2]
    pred_spherical_rescaled = sys.argv[3]
    colors = distinct_colours.get_distinct(4)

    # LOAD FILES

    if kernel == "PS":
        if volume == "small":
            boxsize = 50
            pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                                      "PS_predicted_mass_100_scales_extended_low_mass_range.npy")
            pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                                     "ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")

            if pred_spherical_rescaled is None:
                pass
            else:
                pred_spherical_rescaled = pred_spherical * 9 * np.pi / 2

            ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

        else:
            boxsize = 200
            pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/volume_sharp_k/ALL_PS_predicted_masses.npy")
            pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")

            if pred_spherical_rescaled is None:
                pass
            else:
                pred_spherical_rescaled = pred_spherical * 9 * np.pi / 2

            ic = parameters.InitialConditionsParameters(
                initial_snapshot="/Users/lls/Documents/CODE/sim200/sim200.gadget3",
                final_snapshot="/Users/lls/Documents/CODE/standard200/snapshot_011",
                path="/Users/lls/Documents/CODE/")

        V = boxsize **3
        color = colors[0]
        title = "Press-Schechter"

    else:
        if volume == "small":
            boxsize = 50
            pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/volume_sharp_k/"
                              "ST_predicted_mass_100_scales_extended_low_mass_range.npy")
            pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/correct_growth/"
                                    "ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")

            if pred_spherical_rescaled is None:
                pass
            else:
                pred_spherical_rescaled = np.load("/Users/lls/Documents/CODE/stored_files/hmf/"
                                              "rescale_spherical_to_sk_prediction/ST_rescaled_predicted_masses.npy")

            ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
        else:
            boxsize = 200
            pred_sk = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/volume_sharp_k/"
                              "ALL_ST_predicted_masses.npy")
            pred_spherical = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_ST_predicted_masses.npy")

            if pred_spherical_rescaled is None:
                pass
            else:
                pred_spherical_rescaled = np.load("/Users/lls/Documents/CODE/stored_files/hmf/"
                                              "rescale_spherical_to_sk_prediction/sim200/ALL_ST_predicted_masses.npy")

            ic = parameters.InitialConditionsParameters(
                initial_snapshot="/Users/lls/Documents/CODE/sim200/sim200.gadget3",
                final_snapshot="/Users/lls/Documents/CODE/standard200/snapshot_011",
                path="/Users/lls/Documents/CODE/")

        V = boxsize ** 3
        color = colors[3]
        title = "Sheth-Tormen"

    # CALCULATE HMFS

    # m_min = np.log10(pred_spherical[~np.isnan(pred_spherical)].min())
    # m_max = np.log10(pred_spherical[~np.isnan(pred_spherical)].max())
    m_min = 10
    m_max = 15
    delta_log_M = 0.1

    m_sph, num_sph = ha.get_empirical_number_density_halos(pred_spherical, initial_parameters=ic, boxsize=boxsize,
                                                           log_M_min=m_min, log_M_max=m_max, delta_log_M=delta_log_M)
    m_sk, num_sk = ha.get_empirical_number_density_halos(pred_sk, boxsize=boxsize, initial_parameters=ic,
                                                         log_M_min=m_min, log_M_max=m_max, delta_log_M=delta_log_M)
    m_sk_rescaled, num_sk_rescaled = ha.get_empirical_number_density_halos(pred_spherical_rescaled, boxsize=boxsize,
                                                                           initial_parameters=ic,
                                                                           log_M_min=m_min,
                                                                           log_M_max=m_max, delta_log_M=delta_log_M)

    m_theory, num_theory = ha.get_theory_number_density_halos(kernel, ic, boxsize=boxsize, log_M_min=m_min,
                                                              log_M_max=m_max, delta_log_M=delta_log_M)
    m_bins = np.arange(m_min, m_max, delta_log_M)
    bins = 10 ** m_bins
    delta_m = np.diff(bins)

    # PLOT
    #num_sk_rescaled[11] = 0
    #num_sk[7] = 0

    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    poisson_ps = [np.random.poisson(num_i, 10000) for num_i in num_theory * V]
    vplot2 = ax2.violinplot(poisson_ps, positions=m_theory, widths=delta_m, showextrema=False, showmeans=False, showmedians=False)

    ax2.step(bins[:-1], num_theory * V, where="post", color=color)
    ax2.plot([bins[-2], bins[-1]], [num_theory[-1] * V, num_theory[-1] * V], color=color)
    [b.set_color(color) for b in vplot2['bodies']]

    #ax2.plot(m_theory, num_theory * V, color="k", label="theory")

    ax2.scatter(m_sph, num_sph * V, label="volume sphere", marker="^",color=color)
    ax2.scatter(m_sk, num_sk * V, label="volume sharp-k", marker="o",color=colors[1])
    # ax2.scatter(m_sph * (9 * np.pi /2), num_sph * V, label="horizontal shift only", marker="x", color="k")

    # ax2.scatter(m_sph * (9*np.pi/2), num_sph* V/ (9*np.pi/2), label="analytic expectation", marker="x")
    ax2.scatter(m_sk_rescaled, num_sk_rescaled * V , label="shifted spherical predictions",color="k", marker="x")

    ax2.legend(loc="best", fontsize=15, frameon=False)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # ax2.set_xlim(10**11, 10**14)
    # ax2.set_ylim(10, 2 * 10**3)
    ax2.set_xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
    ax2.set_ylabel("Number of halos")

    ax2.set_title(title + " (" + volume + " box)")

    m_min_plot = ic.initial_conditions['mass'].in_units("Msol h**-1")
    m_max_plot = (np.where(num_theory*V >= 10)[0]).max()
    ax2.set_xlim(m_min_plot[0]*100, 10 ** (np.log10(m_theory[m_max_plot]) + (delta_log_M / 2)))
    ax2.set_ylim(1, 2*10**3)
    plt.tight_layout()

    # plt.savefig("/Users/lls/Desktop/final_hmfs_boxes/ST_rescaled_sph_predictions_w_barrier.png")




    # plt.loglog(m_sph, num_sph, label="volume sphere")
    # plt.loglog(m_sk, num_sk, label="volume sharp-k")
    # plt.loglog(m_sph * (9*np.pi/2), num_sph/ (9*np.pi/2), label="analytic expectation")
    # plt.loglog(m_sk_rescaled, num_sk_rescaled , label="shifted spherical predictions")
    # plt.legend(loc="best", frameon=False)
