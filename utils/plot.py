import matplotlib.pyplot as plt
import numpy as np
import math


# def plot_histogram_mass_halos(all_halos_mass, pure_halos_mass,  bins=50,  probability_threshold):
#     plt.figure(figsize=(16, 5))
#
#     n_all, bins_all, patches_all = plt.hist(np.log10(all_halos_mass), bins=bins, label="all particles",
#                                             normed=False, facecolor='g')
#     n_pure1, bins_pure1, patches_pure1 = plt.hist(np.log10(pure_halos_mass), bins=bins_all, label="pure particles",
#                                                   normed=False, facecolor='b')
#
#     plt.axvline(np.log10(ip_50.halo[0]['mass'].sum()), color='r', label='halo 0-400')
#     plt.axvline(np.log10(ip_50.halo[400]['mass'].sum()), color='r')
#
#     plt.xlabel('Halo mass (Msol) - log scale')
#     plt.ylabel('num particles')
#     plt.title('Histogram of halos for "all"(green) and "pure"(probability > ' + str("%.3f" %  probability_threshold) +
#               ')(blue) particles')
#     plt.legend(loc='best')
#     plt.draw()
#
#     return n_pure1, n_all, bins_all
#
#
# def plot_ratio_pure_all(n_pure, n_all, bins,  probability_threshold, label=None):
#
#     ratio = get_ratio_pure_all(n_pure, n_all)
#
#     # plot
#     plt.figure(figsize=(13,5))
#
#     plt.scatter(bins[1:][n_all!=0], ratio, label=label)
#     plt.plot(bins[1:][n_all!=0], ratio)
#
#     plt.xlabel('Halo mass (log-scale)')
#     plt.ylabel('pure particles/total particles')
#     plt.title('Ratio of pure(probability > ' + str("%.3f" % probability_threshold) +
#               ') particles and total particles for each bin')
#     plt.draw()

def plot_histogram_mass(mass, bins=30, normed=False, label=None):
    n, b, p = plt.hist(mass, bins=bins, normed=normed, label=label)
    plt.draw()


def plot_ratio_histograms_per_bin(ratio, bins, label=None, color=None):
    plt.scatter(bins[1:], ratio, label=label, color=color)
    plt.plot(bins[1:], ratio)
    plt.draw()


def get_log_spaced_bins_flat_distribution(numbers, number_of_bins_init=10, min_halo_number=20):
    log_numbers = np.log10(numbers)

    number_of_bins = number_of_bins_init
    spacing = int(math.ceil(len(numbers) / number_of_bins))

    log_bins = np.zeros((number_of_bins,))
    log_bins[0] = np.sort(log_numbers)[0]
    log_bins[-1] = np.sort(log_numbers)[-1]

    for i in range(1, len(log_bins)-1):
        log_bins[i] = np.sort(log_numbers)[(i * spacing) - 1]

    bins = np.power(10, log_bins)

    unique_numbers = np.unique(numbers)
    n_numbers, b = np.histogram(unique_numbers, bins=bins)

    if (n_numbers < min_halo_number).any():
        bin_remove_index = np.where(n_numbers < min_halo_number)[0][0] + 1
        first_bin_to_remove = bins[bin_remove_index]

        numbers_to_arrange = unique_numbers[unique_numbers > first_bin_to_remove]
        number_bins = int(len(numbers_to_arrange)/min_halo_number)
        bins_large = np.array([numbers_to_arrange[i*min_halo_number] for i in range(1, number_bins+1)])
        bins_large = np.append(bins_large, bins[-1])

        bins = np.concatenate((bins[:bin_remove_index], bins_large))

        n_numbers, b = np.histogram(unique_numbers, bins=bins)

        if n_numbers[-1] < min_halo_number:
            bins = np.delete(bins, -2, 0)
    #
    # elif n_numbers[-1] > 2*n_numbers[-2]:
    #     last_bin = numbers[(numbers>bins[-2]) & (numbers<=bins[-1])]
    #     n_last_bins, bins_last_bin = np.histogram(last_bin, bins=4)
    #     bins = np.append(bins[:-1], bins_last_bin)

    else:
        bins = bins

    return bins

def plot_ellipticity_prolateness_stuff():
    ##### ELLIPTICITY ######

    plt.figure(figsize=(9, 7))

    # ni, bi, pi = plt.hist(subtracted_ell_in[abs(subtracted_ell_in) < 5], histtype="step", color='b', normed=True,
    #                       ls='-', lw=2, label="in",
    #                       bins=30)
    # plt.hist(ell_in[abs(ell_in) < 5], histtype="step", color='b', normed=True, ls='-.', lw=2,
    #          bins=bi)
    plt.hist(noden_ell_in, histtype="step", color='b', normed=True,
             label="in", lw=2,
             bins=30)
    # no, bo, po = plt.hist(subtracted_ell_out[abs(subtracted_ell_out) < 5], histtype="step", color='g', normed=True,
    #                       ls='-', lw=2, label="out",
    #                       bins=30)
    # plt.hist(ell_out[abs(ell_out) < 5], histtype="step", color='g', normed=True, ls='-.', lw=2,
    #          bins=bo)
    plt.hist(noden_ell_out, histtype="step", color='g', normed=True,
             ls='-', lw=2, label="out",
             bins=30)

    # plt.plot(0, label="subtracted-ellipticity", ls='-', color='k')
    # plt.plot(0, label="ellipticity", ls='-.', color='k')
    # plt.plot(0, label="no denominator", ls='--', color='k')
    plt.xlabel(r"$e^{\prime}$ ")
    plt.legend()

    plt.title("Ellipticity (no denominator)")
    plt.tight_layout()
    plt.savefig("/Users/lls/Desktop/eigenvalues_split/ellipticity_no_den.png")

    #### PROLATENESS #####

    plt.figure(figsize=(9, 7))

    # ni, bi, pi = plt.hist(subtracted_prol_in[abs(subtracted_prol_in) < 5], histtype="step", color='b', normed=True,
    #                       ls='-', lw=2, label="in",
    #                       bins=30)
    # plt.hist(prol_in[abs(prol_in) < 5], histtype="step", color='b', normed=True, ls='-.', lw=2,
    #          bins=bi)
    plt.hist(noden_prol_in, histtype="step", color='b', normed=True,  lw=2,
             bins=30, label="in")
    # no, bo, po = plt.hist(subtracted_prol_out[abs(subtracted_prol_out) < 5], histtype="step", color='g', normed=True,
    #                       ls='-', lw=2, label="out",
    #                       bins=30)
    # plt.hist(prol_out[abs(prol_out) < 5], histtype="step", color='g', normed=True, ls='-.', lw=2,
    #          bins=bo)
    plt.hist(noden_prol_out, histtype="step", color='g', normed=True, lw=2,
             bins=30, label="out")

    # plt.plot(0, label="subtracted-prolateness", ls='-', color='k')
    # plt.plot(0, label="prolateness", ls='-.', color='k')
    # plt.plot(0, label="no denominator", ls='--', color='k')
    plt.xlabel(r"$p^{\prime}$ ")
    plt.legend()

    plt.title("Prolateness (no denominator)")
    plt.tight_layout()
    plt.savefig("/Users/lls/Desktop/eigenvalues_split/prolateness_no_den.png")
    plt.clf()

    ###### NO DENOMINATOR ####

    plt.figure(figsize=(9, 7))

    ni, bi, pi = plt.hist(noden_ell_in, histtype="step", color='b', normed=True,
                          ls='-', lw=2,
                          bins=30)
    plt.hist(noden_prol_in, histtype="step", color='b', normed=True, ls='--', lw=2,
             bins=bi)
    no, bo, po = plt.hist(noden_ell_out, histtype="step", color='g', normed=True,
                          ls='-', lw=2,
                          bins=30)
    plt.hist(noden_prol_out, histtype="step", color='g', normed=True, ls='--', lw=2,
             bins=bo)

    plt.plot(0, label="ellipticity", ls='-', color='k')
    plt.plot(0, label="prolateness", ls='--', color='k')
    plt.legend()

    plt.title("Prolateness+Ellipticity (no denominator)")
    plt.tight_layout()
    plt.savefig("/Users/lls/Desktop/eigenvalues_split/no_denominator.png")

    ### individual eigenvalue ###
    for i in [0, 1, 2]:
        plt.figure(figsize=(9, 7))
        plt.hist(eig_in[:, i], histtype="step", color='b', label="in", normed=True, lw=2, bins=20)
        plt.hist(eig_out[:, i], histtype="step", color='g', label="out", normed=True, lw=2, bins=20)
        plt.xlabel(r"\lambda_" + str(i))
        plt.legend()
        plt.savefig("/Users/lls/Desktop/eigenvalues_split/lambda_" + str(i)+ ".png")
        plt.clf()

        plt.figure(figsize=(9, 7))
        plt.hist(subtracted_eig_in[:, i], histtype="step", color='b', label="in", normed=True, lw=2, bins=20)
        plt.hist(subtracted_eig_out[:, i], histtype="step", color='g', label="out", normed=True, lw=2, bins=20)
        plt.xlabel("t_" + str(i))
        plt.legend()
        plt.savefig("/Users/lls/Desktop/eigenvalues_split/density_subtracted_t_" + str(i)+ ".png")
        plt.clf()


# def get_log_spaced_bins_flat_distribution(numbers, number_of_bins_init=10):
#     #log_numbers = np.log10(numbers)
#     sorted_array = np.sort(numbers)
#     bins = np.zeros((978,))
#     bins = np.array([sorted_array[5324 * i] for i in range(977)])
#     assert bins[-1] == numbers.max(), "The last bin did not reach the maximum value of the array"
#     return bins
