import sys
import numpy as np
import pynbody
from scripts.hmf import hmf_theory as ht
from scripts.hmf import hmf_simulation as hs
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
import matplotlib.pyplot as plt
import scipy.stats
import scipy.interpolate
import itertools as it


def get_distribution(number_haloes, coefficient_rescaling):

    m, var, sk, kurtosis = scipy.stats.poisson.stats(number_haloes, moments="mvsk")
    r1 = sk / m
    r2 = kurtosis / m

    if (r1 < 1e-2) and (r2 < 1e-2):
        distr = np.random.normal(number_haloes, np.sqrt(number_haloes), size=10000)
        distr = distr * coefficient_rescaling
        print("Gaussian with num halos " + str(number_haloes))
    else:
        distr = np.random.poisson(number_haloes, 10000)
        distr = distr * coefficient_rescaling
        print("Poisson with num halos " + str(number_haloes))
    return distr


def plot_violin_plot(bins, expectation_per_bin, distr, label="PS", ylabel=r"$ f(\sigma) $",
                     xlabel=r"$\ln \sigma^{-1}$", color="b", log=True):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    color = color
    bins_mid = (bins[1:] + bins[:-1]) / 2
    diff_bins = np.diff(bins)
    plt.ylabel(ylabel)

    if log is True:
        distr = np.log(distr)
        expectation_per_bin = np.log(expectation_per_bin)
        plt.ylabel(r"$ \ln $" + ylabel)

    vplot = axes.violinplot(list(distr), positions=bins_mid, widths=diff_bins, showextrema=False, showmeans=False,
                            showmedians=False)
    [b.set_color(color) for b in vplot['bodies']]
    axes.step(bins[:-1], expectation_per_bin, where="post", color=color, label=label)
    axes.plot([bins[-2], bins[-1]], [expectation_per_bin[-1], expectation_per_bin[-1]], color=color)
    plt.xlabel(xlabel)


def get_fsig_and_distr(mass, numer_of_halos, initial_parameters, diff_ln_sig):
    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    volume = initial_parameters.boxsize_comoving ** 3
    C = mass / (rho_M * volume) / diff_ln_sig

    f_sig = C * numer_of_halos
    distr = np.array([get_distribution(numer_of_halos[i], C[i]) for i in range(len(numer_of_halos))])
    return f_sig, distr


def get_fsig_simulation(log_mass_bins, initial_parameters, diff_ln_sig):
    m_sim, n_halos_sim = hs.get_true_number_halos_per_mass_bins(initial_parameters, log_m_bins=log_mass_bins)

    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    volume = initial_parameters.boxsize_comoving ** 3
    C_sim = m_sim / (rho_M * volume) / diff_ln_sig
    return n_halos_sim * C_sim


def get_variance_function(radius, initial_parameters, z=0, filter=None):
    v = eb.calculate_variance(radius, initial_parameters, z=z, filter=filter)
    f = scipy.interpolate.interp1d(radius, v)
    return f


def get_f_empirical_from_r_predicted(radii_predicitions, initial_parameters, sigma_bins, filter="SK"):
    rho_M = pynbody.analysis.cosmology.rho_M(initial_parameters.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    r_no_nan = radii_predicitions[~np.isnan(radii_predicitions)]

    r_sample = pynbody.array.SimArray(np.linspace(r_no_nan.min(), r_no_nan.max(), 50, endpoint=True))
    r_sample.units = "Mpc h**-1 a"
    if filter == "SK":
        SK = eb.SharpKFilter(initial_parameters.initial_conditions)
        v_function = get_variance_function(r_sample, initial_parameters, z=0, filter=SK)
    else:
        v_function = get_variance_function(r_sample, initial_parameters, z=0)

    v_predicted = v_function(r_no_nan)
    sig_predicted = np.sqrt(v_predicted)

    num_traj, sig_bins_emp = np.histogram(sig_predicted, sigma_bins[::-1])
    m_particle = initial_parameters.initial_conditions['mass'].in_units("Msol h**-1")[0]
    mass_per_bins = num_traj * m_particle

    log_sig_emp = np.log(1 / sig_bins_emp)
    diff_bins_emp = abs(np.diff(log_sig_emp))
    lnsig_mid = (log_sig_emp[1:] + log_sig_emp[:-1]) / 2

    f_emp = mass_per_bins / diff_bins_emp / (rho_M * initial_parameters.boxsize_comoving ** 3)
    return lnsig_mid, f_emp


def get_v_function(radii, initial_parameters):
    r_sample = pynbody.array.SimArray(np.linspace(radii.min(), radii.max(), 50, endpoint=True))
    r_sample.units = "Mpc h**-1 a"
    SK = eb.SharpKFilter(initial_parameters.initial_conditions)
    variance_function = get_variance_function(r_sample, initial_parameters, z=0, filter=SK)
    return variance_function


def split_property_into_8_subboxes(initial_parameters, property):
    l = initial_parameters.boxsize_comoving

    x = initial_parameters.initial_conditions["x"].in_units('Mpc a h**-1')
    y = initial_parameters.initial_conditions["y"].in_units('Mpc a h**-1')
    z = initial_parameters.initial_conditions["z"].in_units('Mpc a h**-1')

    cond_1 = (x > l/2)
    cond_2 = (y > l/2)
    cond_3 = (z > l/2)

    c_1 = property[cond_1 & cond_2 & cond_3]
    c_2 = property[~cond_1 & cond_2 & cond_3]
    c_3 = property[cond_1 & ~cond_2 & cond_3]
    c_4 = property[cond_1 & cond_2 & ~cond_3]
    c_5 = property[~cond_1 & ~cond_2 & cond_3]
    c_6 = property[~cond_1 & cond_2 & ~cond_3]
    c_7 = property[cond_1 & ~cond_2 & ~cond_3]
    c_8 = property[~cond_1 & ~cond_2 & ~cond_3]
    return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8])


def split_property_into_64_subboxes(initial_parameters, property):
    l = initial_parameters.boxsize_comoving

    x = initial_parameters.initial_conditions["x"].in_units('Mpc a h**-1')
    y = initial_parameters.initial_conditions["y"].in_units('Mpc a h**-1')
    z = initial_parameters.initial_conditions["z"].in_units('Mpc a h**-1')

    cond_x = [(x <= l/4), (x > l/4) & (x <= l/2), (x > l/2) & (x <= 3*l/4), (x > 3*l/4) & (x <= l)]
    cond_y = [(y <= l/4), (y > l/4) & (y <= l/2), (y > l/2) & (y <= 3*l/4), (y > 3*l/4) & (y <= l)]
    cond_z = [(z <= l/4), (z > l/4) & (z <= l/2), (z > l/2) & (z <= 3*l/4), (z > 3*l/4) & (z <= l)]

    r_subboxes = []
    for condx in cond_x:
        for condy in cond_y:
            for condz in cond_z:
                r_subboxes.append(property[condx & condy & condz])
    r_subboxes = np.array(r_subboxes)
    return r_subboxes


def get_f_sig_from_bootstrap_sub_volumes(radii_predicted, initial_parameters, mean_den,
                                         sigma_bins, diff_ln_signa, num_bootstrap=100,
                                         variance_function=None, num_sub=8):

    r_no_nan = radii_predicted[~np.isnan(radii_predicted)]
    l = initial_parameters.boxsize_comoving
    m_particle = initial_parameters.initial_conditions['mass'].in_units("Msol h**-1")[0]

    if variance_function is None:
        variance_function = get_v_function(r_no_nan, initial_parameters)

    if num_sub == 8:
        subboxes = split_property_into_8_subboxes(initial_parameters, radii_predicted)
    elif num_sub == 64:
        subboxes = split_property_into_64_subboxes(initial_parameters, radii_predicted)
    else:
        raise Exception("Select either 8 or 64 subboxes")

    f_emp_1 = np.zeros((num_bootstrap, len(diff_ln_signa)))
    for i in range(num_bootstrap):
        a = np.random.choice(range(num_sub), size=num_sub, replace=True)
        boot_r = np.concatenate(subboxes[a]).ravel()
        r_no_nan_boot = boot_r[~np.isnan(boot_r)]

        v_boot = variance_function(r_no_nan_boot)
        sig_boot = np.sqrt(v_boot)

        num_traj_boot, sig_bins_emp = np.histogram(sig_boot, sigma_bins[::-1])
        mass_per_bins_boot = num_traj_boot[::-1] * m_particle
        f_emp_boot = mass_per_bins_boot / diff_ln_signa / (mean_den * (l ** 3))
        f_emp_1[i] = f_emp_boot
        print(f_emp_boot)

    return f_emp_1


if __name__ == "__main__":

    box = sys.argv[1]
    if box == "small" :
        ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
        pred_spherical_growth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/"
                                        "correct_growth/ALL_PS_predicted_masses_1500_even_log_m_spaced.npy")
        rho_M = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
        r_predicted = (pred_spherical_growth / (rho_M * 4 / 3 * np.pi)) ** (1 / 3)

    elif box == "large":
        ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/standard200/standard200.gadget3",
                                                    final_snapshot="/Users/lls/Documents/CODE/standard200/snapshot_011",
                                                    load_final=True, path="/Users/lls/Documents/CODE/")
        pred_spherical_growth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/ALL_PS_predicted_masses.npy")
        rho_M = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
        r_predicted = (pred_spherical_growth / (rho_M * 4 / 3 * np.pi)) ** (1 / 3)

    elif box == "blind":
        ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/reseed50/IC.gadget3",
                                                    final_snapshot="/Users/lls/Documents/CODE/reseed50/snapshot_099",
                                                    load_final=True, path="/Users/lls/Documents/CODE/")
        r_predicted = np.load("/Users/lls/Documents/CODE/reseed50/PS_r_upcrossing.npy")
        rho_M = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")
    else:
        raise InterruptedError

    log_M_min = 10
    log_M_max = 15
    delta_log_M = 0.18

    log_M_bins = np.arange(log_M_min, log_M_max, delta_log_M)
    M_bins = (10**log_M_bins).view(pynbody.array.SimArray)
    M_bins.units = "Msol h^-1"
    v_bins = eb.calculate_variance(M_bins, ic, z=0, cosmology="WMAP5")

    sig_bins = np.sqrt(v_bins)
    lnsig = np.log(1/sig_bins)
    dlnsig = np.diff(lnsig)
    lnsig_mid = (lnsig[1:] + lnsig[:-1])/2

    ####################  SIMULATION  ####################

    f_sig_sim = get_fsig_simulation(log_M_bins, ic, dlnsig)

    #################### PS - Theory ####################

    m_PS, num_halos_PS = ht.theoretical_number_halos(ic, kernel="PS", cosmology="WMAP5", log_M_min=log_M_min,
                                                     log_M_max=log_M_max, delta_log_M=delta_log_M)
    f_sig_PS, distr_PS = get_fsig_and_distr(m_PS, num_halos_PS, ic, dlnsig)


    ###### EMPIRICAL #######

    lnsig_mid_emp, f_empirical = get_f_empirical_from_r_predicted(r_predicted, ic, sig_bins)

    ####### PLOT #######

    # plot_violin_plot(lnsig, f_sig_PS, distr_PS)
    # plt.scatter(lnsig_mid, f_sig_sim, color="k")

    # restrict violin plot to haloes with more than 100 particles
    # and bins with more than 10 halos
    m_min = ic.initial_conditions['mass'].in_units("Msol h^-1")[0] * 100
    restrict_PS_f = np.where((m_PS>=m_min) & (num_halos_PS >=10))[0]
    restrict_PS_bins = np.append(restrict_PS_f, restrict_PS_f[-1] + 1)

    plot_violin_plot(lnsig[restrict_PS_bins], f_sig_PS[restrict_PS_f], distr_PS[restrict_PS_f], color="b", log=True)
    # plt.scatter(lnsig_mid[restrict_PS_f], np.log(f_sig_sim[restrict_PS_f]), color="k", label="sim (" + box + ")")
    plt.scatter(lnsig_mid_emp[::-1][restrict_PS_f], np.log(f_empirical[::-1][restrict_PS_f]), marker="o",
                label="emp (" + box + ")")
    # plt.scatter(ln_m_orig[::-1][restrict_PS_f], np.log(f_orig[::-1][restrict_PS_f]), marker="^", label="emp")

    plt.legend(loc="lower left")
    plt.ylim(-3, 0)
    # plt.xlim(-0.9, 0.45)
    # plt.savefig("/Users/lls/Desktop/f_sigma/f_sig_PS.png")


    ############################  BOOTSTRAP ERRORS  ############################
    #
    # f_boot = get_f_sig_from_bootstrap_sub_volumes(r_predicted, ic, rho_M, sig_bins, dlnsig)
    # mean_bootstrap = np.mean(f_boot, axis=0)
    # std_bootstrap = np.std(f_boot, axis=0)
    #
    # mean_orig = np.mean(f_orig_boot, axis=0)
    # std_orig = np.std(f_orig_boot, axis=0)
    #
    # mean_blind = np.mean(f_blind_boot, axis=0)
    # std_blind = np.std(f_blind_boot, axis=0)
    #
    # plt.plot(lnsig_mid[restrict_PS_f], np.log(f_sig_PS[restrict_PS_f]), color="k", label="theory PS")
    # plt.errorbar(lnsig_mid[restrict_PS_f], np.log(mean_orig[restrict_PS_f]),
    #              yerr=std_orig[restrict_PS_f]/mean_orig[restrict_PS_f],
    #              label="orig", color="b")
    # plt.errorbar(lnsig_mid[restrict_PS_f], np.log(mean_blind[restrict_PS_f]),
    #              yerr=std_blind[restrict_PS_f]/mean_blind[restrict_PS_f],
    #              label="blind", color="g")
    # # plt.scatter(lnsig_mid[restrict_PS_f], np.log(f_empirical[::-1][restrict_PS_f]), s=2)
    # ylabel = r"$ f(\sigma) $"
    # xlabel = r"$\ln \sigma^{-1}$"
    # plt.ylabel(r"$ \ln $" + ylabel)
    # plt.xlabel(xlabel)
    # plt.legend(loc="lower left")
    # plt.ylim(-3, 0)
    #
    #
    #
    #
    #
    # plt.plot(lnsig_mid[restrict_PS_f], np.log(f_sig_PS[restrict_PS_f]), color="k", label="theory PS")
    # plt.errorbar(lnsig_mid[restrict_PS_f], np.log(mean_bootstrap[restrict_PS_f]),
    #              yerr=std_bootstrap[restrict_PS_f]/mean_bootstrap[restrict_PS_f],
    #              label="emp (bootstrap err)", color="b")
    # # plt.scatter(lnsig_mid[restrict_PS_f], np.log(f_empirical[::-1][restrict_PS_f]), s=2)
    # ylabel = r"$ f(\sigma) $"
    # xlabel = r"$\ln \sigma^{-1}$"
    # plt.ylabel(r"$ \ln $" + ylabel)
    # plt.xlabel(xlabel)
    # plt.legend(loc="lower left")
    # plt.ylim(-3, 0)
    #
    # plot_violin_plot(lnsig[restrict_PS_bins], f_sig_PS[restrict_PS_f], distr_PS[restrict_PS_f], color="b", log=True)
    # plt.errorbar(lnsig_mid[restrict_PS_f], np.log(np.mean(f_emp_1, axis=0)[restrict_PS_f]),
    #              yerr=np.std(f_emp_1, axis=0)[restrict_PS_f]/np.mean(f_emp_1, axis=0)[restrict_PS_f],
    #              label="emp (bootstrap err)", color="b")
    # plt.legend(loc="lower left")
    # plt.ylim(-3, 0)
    # plt.savefig("/Users/lls/Desktop/PS_f_sigma_bootstrap_" + box + ".pdf")
    #
    # C = np.cov(f_emp_1.T)
    # c1 = C[restrict_PS_f.min():restrict_PS_f.max() + 1, restrict_PS_f.min():restrict_PS_f.max() +1]
    #
    # # plt.imshow(c1, cmap='magma')
    # # plt.colorbar()
    # # plt.xlabel("Bins")
    # # plt.ylabel("Bins")
    #
    # def chi_squared(data, theory):
    #     covariance = np.cov(data.T)
    #     c_inverse = np.linalg.inv(covariance)
    #     chisq = np.zeros((len(data)))
    #     for i in range(len(data)):
    #         data_i = data[i]
    #         xd = np.mat(data_i - theory)
    #         chisq[i] = xd * np.mat(c_inverse) * xd.T
    #     return np.sum(chisq)






