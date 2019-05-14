import numpy as np
import pynbody
from mlhalos import parameters
from mlhalos import distinct_colours
import matplotlib.pyplot as plt


def get_false_positives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    FPs = ids[labels & ~y_bool]
    return FPs


def get_fraction_FPs_vs_overdensity(false_positives, halo_number, initial_overdensity=200,
                                    f=None, h=None, ids_all=None, num_overdensities=11):

    pynbody.analysis.halo.center(f[h[halo_number].properties['mostboundID']], vel=False)
    f.wrap()
    pynbody.analysis.halo.center(h[halo_number], vel=False)

    r_initial = pynbody.analysis.halo.virial_radius(h[halo_number], overden=initial_overdensity)

    r_FPs = f[false_positives]['r']
    num_FPs = len(np.where(r_FPs <= r_initial)[0])

    # new
    r_all = f[ids_all]['r']
    n_all = len(np.where(r_all <= r_initial)[0])

    overden = [initial_overdensity]
    fraction_false_positives = [num_FPs / n_all]

    den_mean = f.properties["omegaM0"] * pynbody.analysis.cosmology.rho_crit(f, z=0)

    for i in range(num_overdensities - 1):
        r_2 = r_initial * 1.8

        mass = f[pynbody.filt.Sphere(r_2)]['mass'].sum()
        V = (4 / 3) * np.pi * (r_2 ** 3)
        density = mass / V
        overden_2 = density / den_mean

        overden.append(overden_2)

        n_all = len(np.where(r_all <= r_2)[0])
        num_FPs_den = len(np.where(r_FPs <= r_2)[0])

        fraction_false_positives.append(num_FPs_den / n_all)

        r_initial = r_2

    return np.array(overden), np.array(fraction_false_positives)


# Bootstrap method to get errorbars

def do_bootstrap_method(n_fps, bootstrap_number):

    mean_bootstrap = np.zeros((bootstrap_number, 11))
    median_bootstrap = np.zeros((bootstrap_number, 11))
    err = np.zeros((bootstrap_number, 11))

    for i in range(bootstrap_number):
        random_subset = np.random.choice(range(len(n_fps)), int(len(n_fps) * 0.6))

        median_bootstrap[i] = np.median(n_fps[random_subset], axis=0)
        mean_bootstrap[i] = np.mean(n_fps[random_subset], axis=0)
        err[i] = np.std(n_fps[random_subset], axis=0)

    return mean_bootstrap, median_bootstrap, err

if __name__ == "__main__":

    path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"
    th = np.linspace(0, 1, 50)[::-1]
    ids_tested = np.load(path + "tested_ids.npy")
    y_pred_den = np.load(path + "density_only/predicted_den.npy")
    y_true_den = np.load(path + "density_only/true_den.npy")

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    f = ic.final_snapshot
    f.physical_units("kpc")
    h = ic.halo

    Fps_ids_den = get_false_positives(ids_tested, y_pred_den, y_true_den, threshold=th[28])
    ids_in_no_halo = ids_tested[f[ids_tested]['grp']==-1]
    fpd_in_no_halo = Fps_ids_den[np.in1d(Fps_ids_den, ids_in_no_halo)]

    in_halos = 400
    num_overdensities = 11

    for i in range(len(thresholds_plot_fp)):
        Fps_ids_den = get_false_positives(ids_tested, y_pred_den, y_true_den, threshold=th[thresholds_plot_fp[i]])
        ids_in_no_halo = ids_tested[f[ids_tested]['grp'] == -1]
        fpd_in_no_halo = Fps_ids_den[np.in1d(Fps_ids_den, ids_in_no_halo)]
        fps.append(len(fpd_in_no_halo))


    # overden_all = np.zeros((in_halos, num_overdensities))
    # n_tot_all_den = np.zeros((in_halos, num_overdensities))
    # # n_tot_all_den_shear = np.zeros((len(halos), num_overdensities))
    # # halo_mass = np.zeros((len(halos),))
    #
    # for halo_id in range(in_halos):
    #     # halo_number = halos[j]
    #     print("Doing halo " + str(halo_id))
    #     # halo_mass[j] = "%.3g" % h[halo_number]['mass'].sum()
    #     overden_first = 200
    #
    #     overden_all[halo_id], n_tot_all_den[halo_id] = get_fraction_FPs_vs_overdensity(fpd_in_no_halo, halo_id, f=f, h=h,
    #                                                                        ids_all=ids_tested, num_overdensities=11)
    #
    # #Fps_saved = np.load("/Users/lls/Desktop/Fps_th_28_den.npy")


    overden_high = np.load(
        "/Users/lls/Desktop/FPs_shear/threshold_28/FPs_in_no_halo_around_high_mass_halos/overden_all.npy")
    overden_mid = np.load(
        "/Users/lls/Desktop/FPs_shear/threshold_28/FPs_in_no_halo_around_mid_mass_halos/overden_all_71_halos.npy")
    overden_small = np.load("/Users/lls/Desktop/FPs_shear/threshold_28/FPs_in_no_halo_around_small_mass_halos"
                            "/overden_all.npy")
    overden = np.vstack((overden_small, overden_mid, overden_high))
    overden_mean = np.mean(overden, axis=0)

    mid_fps = np.load(
        "/Users/lls/Desktop/FPs_shear/threshold_28/FPs_in_no_halo_around_mid_mass_halos/n_tot_all_den_71_halos.npy")
    high_fps = np.load(
        "/Users/lls/Desktop/FPs_shear/threshold_28/FPs_in_no_halo_around_high_mass_halos/n_tot_all_den.npy")
    small_fps = np.load("/Users/lls/Desktop/FPs_shear/threshold_28/FPs_in_no_halo_around_small_mass_halos/n_tot_all_den"
                        ".npy")
    fps = np.vstack((small_fps, mid_fps, high_fps))

    mean, median, err = do_bootstrap_method(fps, 200)

    # PLOT

    fig, ax = plt.subplots(1, 1, figsize=(6.9, 5.2))
    ind_i = np.argsort(overden_mean)[::-1]
    color = distinct_colours.get_distinct(2)[1]

    ax.errorbar(overden_mean[ind_i], np.mean(mean[:, ind_i], axis=0), yerr=np.std(mean[:, ind_i], axis=0), lw=1.5,
                color=color)
    plt.scatter(overden_mean[ind_i], np.mean(mean[:, ind_i], axis=0),  s=20, color=color)
    # plt.errorbar(overden_mean[ind_i], np.mean(median[:, ind_i], axis=0), yerr=np.std(median[:, ind_i], axis=0),
    #          # color=c[i],
    #          label="median"
    #          )
    # plt.scatter(overden_mean[ind_i], np.mean(median[:, ind_i], axis=0), s=20, marker="x")

    # ax.legend(loc="best", fontsize=15, frameon=False)
    ax.set_xlabel(r"$\rho / \rho_\mathrm{M}$")
    ax.set_ylabel(r"$N_\mathrm{not-in-haloes} / N_\mathrm{all}$", fontsize=17)
    plt.xscale("log")

    plt.axvline(x=200, ls="--", color="k", lw=1.5)
    plt.axvline(x=1, ls="--", color="k", lw=1.5)

    fig.text(0.875, 0.87, "HALO\n(virial)", verticalalignment='top', horizontalalignment='center', fontweight="bold",
             fontsize=15, color="k", transform=ax.transAxes)
    fig.text(0.25, 0.87, "VOID",  verticalalignment='top', horizontalalignment='center', fontweight="bold",
             fontsize=15, color="k", transform=ax.transAxes)

    plt.xlim(0.1, 1000)
    plt.savefig("/Users/lls/Documents/mlhalos_paper/Figure_misclassified_vs_overdensity_2.pdf")
