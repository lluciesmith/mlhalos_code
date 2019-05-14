import numpy as np
from regression.inertia import plot_results as pr
import matplotlib.pyplot as plt
from mlhalos import parameters
from scipy import stats

if __name__ == "__main__":
    path_inertia = "/Users/lls/Documents/mlhalos_files/regression/inertia/"
    path_shear = "/Users/lls/Documents/mlhalos_files/regression/shear/"
    path_density = "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"

    # inertia

    inertia_log_true_mass = np.load(path_inertia + "true_halo_mass.npy")
    bins_plotting = np.linspace(inertia_log_true_mass.min(), inertia_log_true_mass.max(), 15, endpoint=True)

    den_plus_inertia = np.load(path_inertia + "inertia_plus_den/predicted_halo_mass.npy")
    den_plus_inertia_log = np.log10(den_plus_inertia)

    # pf.get_violin_plot(bins_plotting, all_log_predicted_mass, all_log_true_mass, return_mean=False, label_distr="All")
    # plt.savefig(path + "violins.png")

    # shear

    shear_log_true_mass = np.load(path_shear + "true_halo_mass.npy")

    shear_predicted_mass = np.load(path_shear + "predicted_halo_mass.npy")
    shear_log_predicted_mass = np.log10(shear_predicted_mass)

    # density

    den_true = np.load(path_density + "true_halo_mass.npy")
    den_predicted = np.load(path_density + "predicted_halo_mass.npy")
    log_den_true = np.log10(den_true)
    log_den_predicted = np.log10(den_predicted)


    ####################### Radii bins analysis particles only #######################

    testing_ids = np.load(
        "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

    radii_properties_in = np.load("/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_in_ids.npy")
    radii_properties_out = np.load(
        "/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_out_ids.npy")
    fraction = np.concatenate((radii_properties_in[:, 2], radii_properties_out[:, 2]))
    ids_in_halo = np.concatenate((radii_properties_in[:, 0], radii_properties_out[:, 0]))

    # inner radii

    inner_ids = ids_in_halo[fraction < 0.3]
    inner_ids = inner_ids.astype("int")

    ids_inner_tested = np.in1d(testing_ids, inner_ids)

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_inner_tested], inertia_log_true_mass[ids_inner_tested],
                                    shear_log_predicted_mass[ids_inner_tested], inertia_log_true_mass[ids_inner_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="shear",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="inner_particles_vs_shear.pdf"
                                    )
    plt.clf()

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_inner_tested], inertia_log_true_mass[ids_inner_tested],
                                       log_den_predicted[ids_inner_tested], log_den_true[ids_inner_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="density",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="inner_particles_vs_density.pdf"
                                    )
    plt.clf()

    # mid radii

    mid_ids = ids_in_halo[(fraction > 0.3) & (fraction < 0.6)]
    mid_ids = mid_ids.astype("int")

    ids_mid_tested = np.in1d(testing_ids, mid_ids)

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_mid_tested], inertia_log_true_mass[ids_mid_tested],
                                    shear_log_predicted_mass[ids_mid_tested], inertia_log_true_mass[ids_mid_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="shear",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="mid_particles_vs_shear.pdf"
                                    )
    plt.clf()

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_mid_tested], inertia_log_true_mass[ids_mid_tested],
                                       log_den_predicted[ids_mid_tested], log_den_true[ids_mid_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="density",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="mid_particles_vs_density.pdf"
                                    )
    plt.clf()

    # outer

    outer_ids = ids_in_halo[(fraction > 0.6) & (fraction < 1)]
    outer_ids = outer_ids.astype("int")

    ids_outer_tested = np.in1d(testing_ids, outer_ids)

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_outer_tested], inertia_log_true_mass[ids_outer_tested],
                                    shear_log_predicted_mass[ids_outer_tested], inertia_log_true_mass[ids_outer_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="shear",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="outer_particles_vs_shear.pdf"
                                    )
    plt.clf()

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_outer_tested], inertia_log_true_mass[ids_outer_tested],
                                       log_den_predicted[ids_outer_tested], log_den_true[ids_outer_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="density",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="outer_particles_vs_density.pdf"
                                    )
    plt.clf()

    # outer

    nonvir_ids = ids_in_halo[fraction > 1]
    nonvir_ids = nonvir_ids.astype("int")

    ids_nonvir_tested = np.in1d(testing_ids, nonvir_ids)

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_nonvir_tested], inertia_log_true_mass[ids_nonvir_tested],
                                    shear_log_predicted_mass[ids_nonvir_tested], inertia_log_true_mass[ids_nonvir_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="shear",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="nonvir_particles_vs_shear.pdf"
                                    )
    plt.clf()

    pr.violin_plots_density_vs_inertia(den_plus_inertia_log[ids_nonvir_tested], inertia_log_true_mass[ids_nonvir_tested],
                                       log_den_predicted[ids_nonvir_tested], log_den_true[ids_nonvir_tested],
                                    bins_plotting,
                                    label_0="den+inertia",
                                    compare="density",
                                    # color="grey",
                                    path=path_inertia + "inertia_plus_den/",
                                    saving_name="nonvir_particles_vs_density.pdf"
                                    )
    plt.clf()


    #### improvement as a function of radius

    sorted_ids_in_halo = np.sort(ids_in_halo)
    sorted_fraction = fraction[np.argsort(ids_in_halo)]

    fr_tested_ids = sorted_fraction[np.in1d(sorted_ids_in_halo, testing_ids)]
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)

    def frac_diff_predictions_vs_radial_fraction(array_0, array_1, radial_fraction, bins=10, log=True):
        #log_r = np.array([np.inf if x is np.inf else 0 if x == 0 else np.log10(x) for x in radial_fraction])
        # n, b = np.histogram(log_r[~np.isinf(log_r)], bins=bins)

        print(type(bins))
        if type(bins) == "int":
            if log is True:
                b = np.zeros(bins)
                n_s = np.logspace(np.log10(np.unique(radial_fraction)[1]), np.log10(np.unique(radial_fraction)[-2]), bins-1)
                b[1:] = n_s
                print(b)
            else:
                n, b = np.histogram(radial_fraction[~np.isinf(radial_fraction)], bins=bins)
                print(b)
        else:
            b = bins_arranged
            print(b)

        #fractional_diff = (array_0 - array_1)/array_1
        fractional_diff = array_0 - array_1

        mean_each_bin = []
        lower_bound = []
        upper_bound = []
        mid_bins = []
        err_each_bin = []

        for i in range(len(b) - 1):
            indices_each_bin = np.where((radial_fraction >= b[i]) & (radial_fraction < b[i + 1]))[0]
            indices_each_bin = indices_each_bin.astype("int")

            frac_bin = fractional_diff[indices_each_bin]
            if frac_bin.size:
                hist = np.histogram(frac_bin, bins=50)
                hist_dist = stats.rv_histogram(hist)

                c_int_lower, c_int_higher = hist_dist.interval(0.68)
                mean_each_bin.append(np.mean(frac_bin))
                lower_bound.append(c_int_lower)
                upper_bound.append(c_int_higher)

                # mean, var, std = stats.bayes_mvs(frac_bin)
                # mean_each_bin.append(mean.statistic)
                # median_each_bin.append(np.median(frac_bin))
                # lower_bound.append(mean.minmax[0])
                # upper_bound.append(mean.minmax[1])

                mid_bins.append((b[i] + b[i+1])/2)
            else:
                print("pass")

        return np.array(mid_bins), np.array(mean_each_bin), np.array(lower_bound), np.array(upper_bound)

    def bootstrap_mean_diff(array_0, array_1, radial_fraction, bins=10, log=True):
        #log_r = np.array([np.inf if x is np.inf else 0 if x == 0 else np.log10(x) for x in radial_fraction])
        # n, b = np.histogram(log_r[~np.isinf(log_r)], bins=bins)

        print(type(bins))
        if type(bins) == "int":
            if log is True:
                b = np.zeros(bins)
                n_s = np.logspace(np.log10(np.unique(radial_fraction)[1]), np.log10(np.unique(radial_fraction)[-2]), bins-1)
                b[1:] = n_s
                print(b)
            else:
                n, b = np.histogram(radial_fraction[~np.isinf(radial_fraction)], bins=bins)
                print(b)
        else:
            b = bins_arranged
            print(b)

        fractional_diff = (array_0 - array_1)/array_1

        mean_each_bin = []
        error_each_bin = []
        mid_bins = []

        for i in range(len(b) - 1):
            indices_each_bin = np.where((radial_fraction >= b[i]) & (radial_fraction < b[i + 1]))[0]
            indices_each_bin = indices_each_bin.astype("int")

            frac_bin = fractional_diff[indices_each_bin]
            if frac_bin.size:
                m, error = do_bootstrap_method(frac_bin, 100)
                mean_each_bin.append(m)
                error_each_bin.append(error)

                mid_bins.append((b[i] + b[i+1])/2)
            else:
                print("pass")

        return np.array(mid_bins), np.array(mean_each_bin), np.array(error_each_bin)


    bins_arranged = np.concatenate((np.linspace(0, 1, 10, endpoint=False), np.linspace(1, 3, 10)))


    def do_bootstrap_method(array, bootstrap_number):

        mean_bootstrap = np.zeros((bootstrap_number))

        for i in range(bootstrap_number):
            random_subset = np.random.choice(array, len(array))
            mean_bootstrap[i] = np.median(random_subset)

        return np.mean(mean_bootstrap), np.std(mean_bootstrap)

    # single halo

    h_indices = np.concatenate([ic.halo[x]['iord'] for x in range(78)])
    h_indices= ic.halo[99]['iord']
    ind = np.in1d(testing_ids, h_indices)

    mid_bins_in, m_in, l_in, u_in = frac_diff_predictions_vs_radial_fraction(den_plus_inertia_log[ind],
                                                                             log_den_predicted[ind],  fr_tested_ids[
                                                                                 ind], bins=bins_arranged, log=False)
    mid_bins_shear, m_shear, l_shear, u_shear = frac_diff_predictions_vs_radial_fraction(shear_log_predicted_mass[
                                                                                             ind], log_den_predicted[
        ind], fr_tested_ids[ind], bins=bins_arranged, log=False)

    m, me, err = bootstrap_mean_diff(10 ** den_plus_inertia_log[ind], 10 ** log_den_predicted[ind], fr_tested_ids[
        ind], bins=bins_arranged, log=False)

    plt.scatter(mid_bins_in, m_in, label=r"$m_1=$inertia, $m_0=$density", color="b")
    plt.vlines(mid_bins_in, l_in, u_in, color="b")
    #plt.errorbar(m, me, yerr=err)
    plt.scatter(mid_bins_shear, m_shear, color="r", label=r"$m_1=$shear, $m_0=$density")
    plt.vlines(mid_bins_shear, l_shear, u_shear, color="r")

    #plt.axvline((bins_fr[1:] + bins_fr[:-1]) / 2, mean_each_bin)
    plt.axhline(y=0, color="k")
    plt.xlabel(r"$r/\mathrm{r_{vir}}$")
    plt.ylabel(r"$m_1/m_0 - 1$")
    plt.legend(loc="lower right")
    #plt.xscale("log")
    plt.ylim(-0.5,m_in.max() + 0.2)
    plt.xlim(-0.1, 3.1)

    plt.errorbar(mid_bins_in, m_in, yerr=err_in, label=r"$m_1=$inertia, $m_0=$density", color="b")
    plt.errorbar(mid_bins_shear, m_shear, yerr=err_shear, color="r", label=r"$m_1=$shear, $m_0=$density")

    #plt.axvline((bins_fr[1:] + bins_fr[:-1]) / 2, mean_each_bin)
    plt.axhline(y=0, color="k")
    plt.xlabel(r"$r/\mathrm{r_{vir}}$")
    plt.ylabel(r"$m_1/m_0 - 1$")
    plt.legend(loc="lower right")
    #plt.xscale("log")
    plt.ylim(-0.5,m_in.max() + 0.2)
    plt.xlim(-0.1, 3.1)




