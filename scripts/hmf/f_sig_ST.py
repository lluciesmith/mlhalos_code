import sys
import numpy as np
import pynbody
from scripts.hmf import hmf_theory as ht
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
import matplotlib.pyplot as plt
from scripts.hmf import f_sig_PS as fsig


if __name__ == "__main__":
    box = sys.argv[1]

    if box == "small":
        ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
        r_predicted_TH = np.load("/Users/lls/Documents/CODE/stored_files/hmf/upcrossings/ST_r_upcrossing_TH_sigma.npy")
        r_predicted_SK = np.load("/Users/lls/Documents/CODE/stored_files/hmf/upcrossings/ST_r_upcrossing_SK_sigma.npy")

    elif box == "large":
        ic = parameters.InitialConditionsParameters(
            initial_snapshot="/Users/lls/Documents/CODE/standard200/standard200.gadget3",
            final_snapshot="/Users/lls/Documents/CODE/standard200/snapshot_011",
            load_final=True, path="/Users/lls/Documents/CODE/")
        r_predicted_TH = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/upcrossings/predicted_radii_TH_sig.npy")
        r_predicted_SK = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/upcrossings/predicted_radii_SK_sig.npy")

    elif box == "blind":
        ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/reseed50/IC.gadget3",
                                                    final_snapshot="/Users/lls/Documents/CODE/reseed50/snapshot_099",
                                                    load_final=True, path="/Users/lls/Documents/CODE/")
        r_predicted_SK = np.load("/Users/lls/Documents/CODE/reseed50/ST_r_upcrossing_SK_sigma.npy")
        r_predicted_TH = np.load("/Users/lls/Documents/CODE/reseed50/ST_r_upcrossing_TH_sigma.npy")
    else:
        raise RuntimeError

    log_M_min = 10
    log_M_max = 15
    delta_log_M = 0.15

    log_M_bins = np.arange(log_M_min, log_M_max, delta_log_M)
    M_bins = (10**log_M_bins).view(pynbody.array.SimArray)
    M_bins.units = "Msol h^-1"
    v_bins = eb.calculate_variance(M_bins, ic, z=0, cosmology="WMAP5")

    sig_bins = np.sqrt(v_bins)
    lnsig = np.log(1/sig_bins)
    dlnsig = np.diff(lnsig)
    lnsig_mid = (lnsig[1:] + lnsig[:-1])/2

    ####################  SIMULATION  ####################

    f_sig_sim = fsig.get_fsig_simulation(log_M_bins, ic, dlnsig)

    ###################  THEORY  ####################

    m_ST, num_halos_ST = ht.theoretical_number_halos(ic, kernel="ST", cosmology="WMAP5", log_M_min=log_M_min,
                                                     log_M_max=log_M_max, delta_log_M=delta_log_M)
    f_sig_ST, distr_ST = fsig.get_fsig_and_distr(m_ST, num_halos_ST, ic, dlnsig)

    ###################  EMPIRICAL  ####################

    lnsig_mid_emp_TH, f_empirical_TH = fsig.get_f_empirical_from_r_predicted(r_predicted_TH, ic, sig_bins)
    lnsig_mid_emp_SK, f_empirical_SK = fsig.get_f_empirical_from_r_predicted(r_predicted_SK, ic, sig_bins)

    ###################  PLOT  ####################

    # fsig.plot_violin_plot(lnsig, f_sig_ST, distr_ST, color="g", label="ST")
    # plt.scatter(lnsig_mid, np.log(f_sig_sim), color="k")
    # plt.yscale("log")
    #plt.ylim(0,0.5)


    # restrict violin plot to haloes with more than 100 particles
    # and bins with more than 10 halos
    m_min = ic.initial_conditions['mass'].in_units("Msol h^-1")[0] * 100
    restrict_ST_f = np.where((m_ST>=m_min) & (num_halos_ST >=10))[0]
    restrict_ST_bins = np.append(restrict_ST_f, restrict_ST_f[-1] + 1)

    # fsig.plot_violin_plot(lnsig[restrict_ST_bins], f_sig_ST[restrict_ST_f], distr_ST[restrict_ST_f], color="g", label="ST")
    # # plt.scatter(lnsig_mid[restrict_ST_f], np.log(f_sig_sim[restrict_ST_f]), color="k", label="sim")
    # #plt.scatter(lnsig_mid_emp_TH[::-1][restrict_ST_f], np.log(f_empirical_TH[::-1][restrict_ST_f]), marker="^",
    # #            label="emp(TH)", color="g")
    # plt.scatter(lnsig_mid_emp_SK[::-1][restrict_ST_f], np.log(f_empirical_SK[::-1][restrict_ST_f]), marker="o",
    #             label="emp (" + box + ")", color="g")
    #
    # plt.legend(loc="lower left")
    # plt.ylim(-3,0)
    # plt.xlim(-1,-0.2)
    # plt.savefig("/Users/lls/Desktop/f_sig_ST_large_only_SK.pdf")
    # plt.savefig("/Users/lls/Desktop/f_sig_ST_.png")


    ############################  BOOTSTRAP ERRORS  ############################

    r_predicted = r_predicted_SK
    r_no_nan = r_predicted[~np.isnan(r_predicted)]

    r_sample = pynbody.array.SimArray(np.linspace(r_no_nan.min(), r_no_nan.max(), 50, endpoint=True))
    r_sample.units = "Mpc h**-1 a"
    SK = eb.SharpKFilter(ic.initial_conditions)
    v_function = fsig.get_variance_function(r_sample, ic, z=0, filter=SK)
    rho_M = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol Mpc^-3 h^2 a^-3")

    m_particle = ic.initial_conditions['mass'].in_units("Msol h**-1")[0]
    l = 50

    x = ic.initial_conditions["x"].in_units('Mpc a h**-1')
    y = ic.initial_conditions["y"].in_units('Mpc a h**-1')
    z = ic.initial_conditions["z"].in_units('Mpc a h**-1')

    cond_1 = (x > l/2)
    cond_2 = (y > l/2)
    cond_3 = (z > l/2)

    c_1 = r_predicted[cond_1 & cond_2 & cond_3]
    c_2 = r_predicted[~cond_1 & cond_2 & cond_3]
    c_3 = r_predicted[cond_1 & ~cond_2 & cond_3]
    c_4 = r_predicted[cond_1 & cond_2 & ~cond_3]
    c_5 = r_predicted[~cond_1 & ~cond_2 & cond_3]
    c_6 = r_predicted[~cond_1 & cond_2 & ~cond_3]
    c_7 = r_predicted[cond_1 & ~cond_2 & ~cond_3]
    c_8 = r_predicted[~cond_1 & ~cond_2 & ~cond_3]

    subboxes = np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8])

    f_emp_1 = np.zeros((1000, len(lnsig_mid)))
    for i in range(1000):
        a = np.random.choice(range(8), size=8, replace=True)
        boot_r = np.concatenate(subboxes[a]).ravel()
        r_no_nan_boot = boot_r[~np.isnan(boot_r)]

        v_boot = v_function(r_no_nan_boot)
        sig_boot = np.sqrt(v_boot)

        num_traj_boot, sig_bins_emp = np.histogram(sig_boot, sig_bins[::-1])
        mass_per_bins_boot = num_traj_boot[::-1] * m_particle
        f_emp_boot = mass_per_bins_boot / dlnsig / (rho_M * ic.boxsize_comoving ** 3)
        f_emp_1[i] = f_emp_boot

    # plt.plot(lnsig_mid[restrict_ST_f], np.log(f_sig_ST[restrict_ST_f]))
    # plt.errorbar(lnsig_mid[restrict_ST_f], np.log(np.mean(f_emp_1, axis=0)[restrict_ST_f]),
    #              yerr=np.std(f_emp_1, axis=0)[restrict_ST_f]/np.mean(f_emp_1, axis=0)[restrict_ST_f],
    #              label="emp", color="g")
    # plt.scatter(lnsig_mid[restrict_ST_f], np.log(f_empirical_SK[::-1][restrict_ST_f]))
    # plt.ylim(-3, 0)

    fsig.plot_violin_plot(lnsig[restrict_ST_bins], f_sig_ST[restrict_ST_f], distr_ST[restrict_ST_f], color="g",
                          label="ST")
    plt.errorbar(lnsig_mid[restrict_ST_f], np.log(np.mean(f_emp_1, axis=0)[restrict_ST_f]),
                 yerr=np.std(f_emp_1, axis=0)[restrict_ST_f]/np.mean(f_emp_1, axis=0)[restrict_ST_f],
                 label="emp (bootstrap err)", color="g")
    plt.legend(loc="lower left")
    plt.ylim(-3, 0)
    plt.savefig("/Users/lls/Desktop/ST_f_sigma_bootstrap.pdf")

    def covariance_matrix(x_i, x_j):
        x_i_exp = np.mean(x_i, axis=0)
        c = (a/N-1)


