import numpy as np
import sys

import scripts.hmf.hmf_theory

sys.path.append("/Users/lls/Documents/mlhalos_code")
from scripts.hmf import hmf_empirical as hmf_emp
from scripts.hmf import hmf_theory
from scripts.hmf import hmf_simulation as hmf_sim
from mlhalos import parameters
import matplotlib.pyplot as plt
from mlhalos import distinct_colours


def get_empirical_number_density_halos(predicted_mass, initial_parameters, boxsize=50, log_M_min=10, log_M_max=15,
                                       delta_log_M=0.1):
    m_bins = np.arange(log_M_min, log_M_max, delta_log_M)
    m_mid = (m_bins[1:] + m_bins[:-1])/2
    num_halos = hmf_emp.get_empirical_number_halos(predicted_mass, initial_parameters, log_M_min=log_M_min,
                                                   log_M_max=log_M_max, delta_log_M=delta_log_M)
    volume = boxsize ** 3
    num_density = num_halos/volume
    return 10**m_mid, num_density


def get_theory_number_density_halos(kernel, initial_parameters, boxsize=50, log_M_min=10, log_M_max=15,
                                    delta_log_M=0.1):
    mass, num_halos = hmf_theory.theoretical_number_halos(initial_parameters, kernel=kernel, log_M_min=log_M_min,
                                                                      log_M_max=log_M_max, delta_log_M=delta_log_M)
    volume = boxsize ** 3
    num_density = num_halos/volume
    return mass, num_density


def get_simulation_number_density_halos(initial_parameters, boxsize=50, log_m_bins=None):
    if log_m_bins is None:
        log_m_bins = np.arange(10, 15, 0.1)

    m_true, num_true = hmf_sim.get_true_number_halos_per_mass_bins(initial_parameters, log_m_bins)
    num_den_true = num_true/(boxsize**3)
    return m_true, num_den_true


if __name__ == "__main__":

    # Initial conditions for small and larger boxes

    initial_snapshot = "/Users/lls/Documents/CODE/larger_sim/sim200.gadget3"

    initial_parameters_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                                    path="/Users/lls/Documents/CODE/")
    initial_parameters_50 = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")

    # Load predicted masses PS/ST for smaller and larger boxes

    predicted_mass_PS_large = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")
    predicted_mass_ST_large = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_ST_predicted_masses.npy")

    predicted_mass_PS_smaller = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k"
                                        "/ALL_predicted_masses_1500_even_log_m_spaced.npy")
    predicted_mass_ST_smaller = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k"
                                        "/ALL_ST_predicted_masses_1500_even_log_m_spaced.npy")

    # Get predicted number density

    PS_large_box = get_empirical_number_density_halos(predicted_mass_PS_large, initial_parameters_200, boxsize=200)
    ST_large_box = get_empirical_number_density_halos(predicted_mass_ST_large, initial_parameters_200, boxsize=200)

    PS_small_box = get_empirical_number_density_halos(predicted_mass_PS_smaller, initial_parameters_50, boxsize=50)
    ST_small_box = get_empirical_number_density_halos(predicted_mass_ST_smaller, initial_parameters_50, boxsize=50)


    # Theoretical with violin plot

    m_PS, num_PS = get_theory_number_density_halos("PS", initial_parameters_50, boxsize=50)
    m_ST, num_ST = get_theory_number_density_halos("ST", initial_parameters_50, boxsize=50)

    m_PS_check, num_PS_check = get_theory_number_density_halos("PS", initial_parameters_200, boxsize=200)
    m_ST_check, num_ST_check = get_theory_number_density_halos("ST", initial_parameters_200, boxsize=200)
    assert (num_PS == num_PS_check).all()
    assert (num_ST == num_ST_check).all()


    # Simulation hmf

    m_true_50, num_true_50 = get_simulation_number_density_halos(initial_parameters_50, boxsize=50, log_m_bins=None)


# Plots


def plot_curves_PS():
    m = np.arange(10, 15, 0.1)
    m_mid = (m[1:] + m[:-1]) / 2
    plt.loglog(10**m_mid, PS_large_box, color="b", label="PS", lw=2)
    plt.loglog(10**m_mid, PS_small_box, color="b", label="PS")
    plt.loglog(m_PS, num_PS, color="b", ls="--")
    plt.loglog(m_true_50, num_true_50, color="k")
    plt.legend(loc="best")


def plot_curves_ST():
    m = np.arange(10, 15, 0.1)
    m_mid = (m[1:] + m[:-1]) / 2

    plt.loglog(10**m_mid, ST_large_box , color="g", label="ST (large)", lw=2)
    plt.loglog(10**m_mid, ST_small_box, color="g", label="ST (small)")
    plt.loglog(m_ST, num_ST, color="g", ls="--")
    plt.loglog(m_true_50, num_true_50, color="k")

    plt.legend(loc="best")

def plot_step_and_dots_PS():
    m_bins = 10 ** np.arange(10, 15, 0.1)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    colors = distinct_colours.get_distinct(4)

    axes.step(m_bins[:-1], num_PS, where="post", color=colors[0], label="PS")
    axes.plot([m_bins[-2], m_bins[-1]], [num_PS[-1], num_PS[-1]], color=colors[0])

    plt.scatter(m_true_50, num_true_50, color="k", label="simulation")

    plt.scatter(10**m_mid, PS_large_box, marker="^", color=colors[0], label="empirical (large)", s=60, alpha=1)
    plt.scatter(10 ** m_mid, PS_small_box, marker="x", color=colors[0], label="empirical (small)", s=60, alpha=1)

    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Number density of halos")
    plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")


def plot_step_and_dots_ST():
    m_bins = 10 ** np.arange(10, 15, 0.1)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    colors = distinct_colours.get_distinct(4)

    axes.step(m_bins[:-1], num_ST, where="post", color=colors[0], label="ST")
    axes.plot([m_bins[-2], m_bins[-1]], [num_ST[-1], num_ST[-1]], color=colors[0])

    plt.scatter(m_true_50, num_true_50, color="k", label="simulation")

    plt.scatter(10 ** m_mid, ST_large_box, marker="^", color=colors[0], label="empirical (large)", s=60, alpha=1)
    plt.scatter(10 ** m_mid, ST_small_box, marker="x", color=colors[0], label="empirical (small)", s=60, alpha=1)

    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Number density of halos")
    plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")


def violin_plot_PS():
    m_bins = 10 ** np.arange(10, 15, 0.1)

    abs_num_PS = num_PS * (50**3)
    poisson_PS = [np.random.poisson(num_i, 10000) for num_i in abs_num_PS]

    delta_m = np.diff(m_bins)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    colors = distinct_colours.get_distinct(4)

    vplot = axes.violinplot(poisson_PS, positions=m_PS, widths=delta_m, showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(colors[0]) for b in vplot['bodies']]

    axes.step(m_bins[:-1], abs_num_PS, where="post", color=colors[0], label="PS")
    axes.plot([m_bins[-2], m_bins[-1]], [abs_num_PS[-1], abs_num_PS[-1]], color=colors[0])

    plt.scatter(m_true_50, num_true_50*(50**3), color="k", label="simulation")

    plt.scatter(10**m_mid, PS_large_box*(50**3), marker="^", color=colors[0], label="empirical (large)", s=60,
                alpha=1)
    plt.scatter(10 ** m_mid, PS_small_box*(50**3), marker="x", color=colors[0], label="empirical (small)", s=60,
                alpha=1)

    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Number of halos")
    plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
    plt.xlim(5 * 10 ** 10, 10 ** 14)
    plt.savefig("/Users/lls/Desktop/num_halos_small_vs_large_box_PS.png")


def get_theory_violin_plot(log_mass_bins, number_halos, emp_small, emp_large, label="PS", restricted_mass_range=None):

    if restricted_mass_range is not None:
        number_halos = number_halos[restricted_mass_range]
        ind = np.where(restricted_mass_range)[0]
        ind_bins = np.append(ind, ind[-1] + 1)
        log_mass_bins = log_mass_bins[ind_bins]

    poisson = [np.random.poisson(num_i, 10000) for num_i in (50**3)*number_halos]
    delta_m = np.diff(10**log_mass_bins)
    mass = 10 ** ((log_mass_bins[1:] + log_mass_bins[:-1]) / 2)


    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    colors = distinct_colours.get_distinct(4)

    if label == "PS":
        c = colors[0]
    else:
        c = colors[3]

    vplot = axes.violinplot(poisson, positions=mass[restricted_mass_range],
                            widths=delta_m[restricted_mass_range],
                            showextrema=False,
                            showmeans=False,
                            showmedians=False)
    [b.set_color(c) for b in vplot['bodies']]


    axes.step(10**log_mass_bins[:-1], (50**3)*number_halos, where="post", color=c, label=label)
    axes.plot([10**log_mass_bins[-2], 10**log_mass_bins[-1]],
              [(50**3)* number_halos[-1], (50**3)*number_halos[-1]], color=c)

    plt.scatter(mass, emp_large[restricted_mass_range], color=c, label="large")
    plt.scatter(mass, emp_small[restricted_mass_range], marker="^", color=c, label="small")
    # plt.scatter(mass[restricted_mass_range], num_sim[restricted_mass_range], color="k", label="sim")

    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Number of halos")
    plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
    plt.xlim(10**log_mass_bins.min(), 10**log_mass_bins.max())



def violin_plot_ST(PS_theory, ST_theory, bins):
    m_bins = 10 ** np.arange(10, 15, 0.1)

    abs_num_ST = num_ST * (50**3)
    poisson_ST = [np.random.poisson(num_i, 10000) for num_i in abs_num_ST]

    delta_m = np.diff(m_bins)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    colors = distinct_colours.get_distinct(4)

    vplot = axes.violinplot(poisson_ST, positions=m_ST, widths=delta_m, showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(colors[3]) for b in vplot['bodies']]
    # [b.set_color("blue") for b in [vplot['cmaxes'], vplot['cbars'], vplot['cmins']]]

    axes.step(m_bins[:-1], abs_num_ST, where="post", color=colors[3], label="ST")
    axes.plot([m_bins[-2], m_bins[-1]], [abs_num_ST[-1], abs_num_ST[-1]], color=colors[3])

    plt.scatter(m_true_50, num_true_50*(50**3), color="k", label="simulation")

    plt.scatter(10**m_mid, ST_large_box*(50**3), marker="^", color=colors[3], label="empirical (large)", s=60,
                alpha=1)
    plt.scatter(10 ** m_mid, ST_small_box*(50**3), marker="x", color=colors[3], label="empirical (small)", s=60,
                alpha=1)

    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Number of halos")
    plt.xlabel(r"$ \mathrm{M} [\mathrm{M}_{\odot} \mathrm{h}^{-1}]$")
    plt.xlim(5 * 10 ** 10, 10 ** 14)
    plt.xlim(m_bins[9], m_bins[-17])


