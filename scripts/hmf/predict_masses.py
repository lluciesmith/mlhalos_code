import sys
import numpy as np
#sys.path.append("/Users/lls/Documents/mlhalos_code")
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import window
from mlhalos import parameters
from scripts.ellipsoidal import ellipsoidal_barrier as eb
import importlib
importlib.reload(eb)
import matplotlib.pyplot as plt
import pynbody


############################ COLLAPSE BARRIERS AND FIRST UPCROSSINGS ############################


def get_threshold_barrier(masses, initial_parameters, barrier="ST", cosmology="WMAP5", a=0.707, z=99,
                          delta_sc_0=1.686, growth=None, filter=None):
    if barrier == "ST":
        b = eb.ellipsoidal_collapse_barrier(masses, initial_parameters, beta=0.5, gamma=0.6, a=a, z=z,
                                            cosmology=cosmology, output="rho/rho_bar", delta_sc_0=delta_sc_0,
                                            filter=filter)

    elif barrier == "spherical":
        delta_sc = eb.get_spherical_collapse_barrier(initial_parameters, z=z, delta_sc_0=delta_sc_0,
                                                     output="rho/rho_bar", growth=growth)
        b = np.tile([delta_sc], len(masses))

    else:
        raise NameError("Wrong barrier - select either ST or spherical")

    return b


def record_mass_first_upcrossing(threshold_barrier, filter_masses, trajectories=None,
                                 halo_min=None):
    """
    This function excludes all particles that are NOT in halos according to SUBFIND.
    """
    # This line excludes all particles that are NOT in halos according to SUBFIND.
    # traj = get_trajectories_of_particles_in_halos(initial_conditions, trajectories, halo_min)

    traj = trajectories
    upcrossing = np.zeros((len(traj),))

    for i in range(len(traj)):
            upcrossing[i] = trajectory_first_upcrossing(traj[i], threshold_barrier, filter_masses)

    upcrossing = upcrossing.view(pynbody.array.SimArray)
    upcrossing.units = filter_masses.units
    return upcrossing


def trajectory_first_upcrossing(trajectory, threshold, smoothing_masses):
    above_threshold = np.where(trajectory >= threshold)[0]

    if above_threshold.size == 0:
        first_upcrossing = np.nan
    else:
        first_upcrossing_index = above_threshold[-1]
        first_upcrossing = smoothing_masses[first_upcrossing_index]
    return first_upcrossing


def get_predicted_analytic_mass(masses, initial_parameters, barrier="ST", cosmology="WMAP5", a=0.707,
                                trajectories=None, delta_sc_0=1.686, growth=None, filter=None):
    barrier_value = get_threshold_barrier(masses, initial_parameters, barrier=barrier, cosmology=cosmology, a=a,
                                          delta_sc_0=delta_sc_0, growth=growth, filter=filter)

    if len(trajectories.shape) != 1:
        mass_first_upcrossing = record_mass_first_upcrossing(barrier_value, masses,
                                                             trajectories=trajectories,
                                                             halo_min=None)
    else:
        mass_first_upcrossing = trajectory_first_upcrossing(trajectories, barrier_value, masses)
        #mass_first_upcrossing = mass_first_upcrossing.view(pynbody.array.SimArray)
        #mass_first_upcrossing.units = masses.units
    return mass_first_upcrossing


############################ FUNCTIONS ON PARTICLES IN HALOS ONLY ############################


def get_particles_in_halos(initial_conditions, halo_min=None):
    if halo_min is None:
        halo_min = 16320

    snapshot = initial_conditions.final_snapshot
    particles_in_halos = snapshot['iord'][(snapshot['grp'] <= halo_min) & (snapshot['grp'] >= 0)]
    return particles_in_halos


def get_trajectories_of_particles_in_halos(initial_conditions, trajectories=None, halo_min=None):
    if trajectories is None:
        trajectories = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/density_trajectories.npy")

    particles_in_halos = get_particles_in_halos(initial_conditions, halo_min)
    if len(trajectories) > len(particles_in_halos):
        traj = trajectories[particles_in_halos, :]
    else:
        traj = trajectories
    return traj


def get_true_mass_particles_in_halos(initial_conditions, halo_min=None):
    snapshot = initial_conditions.final_snapshot
    if halo_min is None:
        halo_min = 16320

    try:
        halo_mass_particles = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy")
        print("Loaded saved halo mass particles file")
        true_mass = halo_mass_particles[(snapshot['grp'] <= halo_min) & (snapshot['grp'] >= 0)]

    except IOError:
        particles = get_particles_in_halos(initial_conditions, halo_min=halo_min)
        halos = snapshot[particles]['grp']
        true_mass = np.array([initial_conditions.halo[n]['mass'].sum() for n in halos])
    return true_mass


def from_radius_predict_mass(smoothing_r, initial_params, traject=None, barrier="spherical"):
    r_comoving = radius_comoving_h_units(smoothing_r, initial_params.initial_conditions)
    smoothing_m = pynbody_r_to_m(r_comoving, initial_params.initial_conditions)
    pred_mass = get_predicted_analytic_mass(smoothing_m, initial_params, barrier=barrier, cosmology="WMAP5",
                                            trajectories=traject)
    return pred_mass


def get_position_particles_belonging_to_halo(halo_num, particles, initial_parameters):
    if isinstance(halo_num, int):
        pos = initial_parameters.final_snapshot[particles]['grp'] == halo_num
    else:
        pos = initial_parameters.final_snapshot[particles]['grp'] == halo_num[0]
        for i in range(1, len(halo_num)):
            pos_i = initial_parameters.final_snapshot[particles]['grp'] == halo_num[i]
            pos = pos | pos_i
    return pos


def pynbody_r_to_m(radius, snapshot):
    """ Input should be radius in units [Mpc a h^-1]"""
    th = pynbody.analysis.hmf.TophatFilter(snapshot)
    m = (th.R_to_M(radius)).view(pynbody.array.SimArray)
    m.units = "Msol h^-1"
    return m


def pynbody_m_to_r(mass, snapshot, units="comoving"):
    """ Input should be mass in units [Msol h^-1]"""
    th = pynbody.analysis.hmf.TophatFilter(snapshot)
    r = th.M_to_R(mass)
    if units == "comoving":
        r.units = "Mpc a h^-1"
    elif units == "physical":
        r = r.in_units("Mpc", **snapshot.conversion_context())
    else:
        raise NameError("Select either comoving or physical units")
    return r


def radius_comoving_h_units(radius, snapshot):
    r_a_h = radius/snapshot.properties['a'] * snapshot.properties['h']
    r_a_h = r_a_h.view(pynbody.array.SimArray)
    r_a_h.units = "Mpc a h^-1"
    return r_a_h


############################ SCATTER PLOTS PREDICTED MASS VS TRUE ############################


def scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_mass, ellipsoidal_mass, true_mass, particles,
                                                        random_scatter):

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

    ax1.scatter(np.log10(spherical_mass[particles]) + random_scatter,
                np.log10(true_mass[particles]) + random_scatter, alpha=0.2, s=5)
    ax2.scatter(np.log10(ellipsoidal_mass[particles]) + random_scatter,
                np.log10(true_mass[particles]) + random_scatter, alpha=0.2, s=5)


    min_x = min([np.log10(spherical_mass[particles].min()), np.log10(ellipsoidal_mass[particles]).min()])
    max_x = max([np.log10(spherical_mass[particles].max()), np.log10(ellipsoidal_mass[particles]).max()])

    ax1.set_xlim(min_x, max_x)
    ax2.set_ylim(np.log10(true_mass[particles]).min(), np.log10(true_mass[particles]).max())

    x = np.arange(min_x, max_x)
    ax1.plot(x, x, color="k", ls="-")
    ax2.plot(x, x, color="k", ls="-")

    fig.subplots_adjust(wspace=0)
    # plt.xscale("log")
    # plt.yscale("log")

    ax1.set_ylabel("$\log_{10}\mathrm{M}_{\mathrm{halo}} [ \mathrm{M}_\odot ]$")
    ax1.set_xlabel("$\log_{10}\mathrm{M}_{\mathrm{spherical}} [ \mathrm{M}_\odot ]$")
    ax2.set_xlabel("$\log_{10}\mathrm{M}_{\mathrm{ellipsoidal}} [\mathrm{M}_\odot ]$")



def get_histogram_predicted_masses(halo_num, initial_parameters, particles_predicted=None, bins_num=20, normed=True,
                                   predicted_new_filt=None, predicted_250=None,
                                   pred_1500=None):
    if particles_predicted is None:
        particles_predicted = get_particles_in_halos(initial_parameters)
    #
    if predicted_new_filt is None:
        predicted_new_filt = np.load("/Users/lls/Documents/CODE/stored_files/hmf/pred_Msol_h_new_filt_bound_030.npy")

    if predicted_250 is None:
        predicted_250 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_radii_predicted_masses_PS.npy")
        #predicted_250 = predicted_250 * initial_parameters.final_snapshot['h']
    if pred_1500 is None:
        pred_1500 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sharp_trajectories_1500_predicted_masses.npy")
        #pred_1500 = pred_1500 * initial_parameters.final_snapshot['h']

    halo_pos = get_position_particles_belonging_to_halo(halo_num, particles_predicted, initial_parameters)

    #halo_pos = np.where(initial_parameters.final_snapshot[particles_predicted]['grp'] == halo_num)[0]
    h_predicted_250 = predicted_250[halo_pos]
    h_pred_1500 = pred_1500[halo_pos]
    h_new_filt = predicted_new_filt[halo_pos]

    bins = 10**np.linspace(10, 15, bins_num)

    plt.hist(h_predicted_250[~np.isnan(h_predicted_250)], bins=bins, histtype="step", normed=normed,
             label="250 logM")
    plt.hist(h_pred_1500[~np.isnan(h_pred_1500)], bins=bins, histtype="step", normed=normed, label="1500 r")
    plt.hist(h_new_filt[~np.isnan(h_new_filt)], bins=bins, histtype="step", normed=normed, label="new "
                                                                                                           "filtering")
    if not isinstance(halo_num, int):
        for i in halo_num:
            plt.axvline(x=initial_parameters.halo[i]['mass'].sum() * 0.701, color="k", ls="--")
    else:
        plt.axvline(x=initial_parameters.halo[halo_num]['mass'].sum()*0.701, color="k", ls="--")
    plt.legend(loc="best")
    plt.xscale("log")


def get_histogram_predicted_masses_2(halo_num, initial_parameters, particles_predicted=None, bins_num=20, predicted_50=None,
                                   predicted_250=None, predicted_1500=None, normed=False):

    if particles_predicted is None:
        particles_predicted = get_particles_in_halos(initial_parameters)

    if predicted_50 is None:
        predicted_50 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k"
                                "/pred_mass_50_smoothing.npy")

    if predicted_250 is None:
        predicted_250 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k/pred_mass_250_smoothing"
                               ".npy")
    if predicted_1500 is None:
        predicted_1500 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k/pred_mass_1500_smoothing"
                                ".npy")
    halo_pos = get_position_particles_belonging_to_halo(halo_num, particles_predicted, initial_parameters)

    #halo_pos = np.where(initial_parameters.final_snapshot[particles_predicted]['grp'] == halo_num)[0]
    h_predicted_250 = predicted_250[halo_pos]
    h_predicted_50 = predicted_50[halo_pos]
    h_predicted_1500 = predicted_1500[halo_pos]

    bins = 10**np.linspace(10, 15, bins_num)

    a = h_predicted_250[~np.isnan(h_predicted_250)]
    b = h_predicted_50[~np.isnan(h_predicted_50)]
    c = h_predicted_1500[~np.isnan(h_predicted_1500)]

    plt.figure(figsize=(9, 6))
    # plt.hist(h_predicted_250[~np.isnan(h_predicted_250)], bins=bins, histtype="bar", normed=normed, alpha=0.5,
    #          label="250 radii")
    # plt.hist(h_predicted_50[~np.isnan(h_predicted_50)], bins=bins, histtype="bar",alpha=0.5,
    #          normed=normed, label="50 radii")
    # plt.hist(h_predicted_1500[~np.isnan(h_predicted_1500)], bins=bins, histtype="bar",alpha=0.5,
    #          normed=normed, label="1500 radii")
    plt.hist((b, a, c), bins=bins, histtype="bar", normed=normed,
             label=["50 radii", "250 radii", "1500 radii"])

    if not isinstance(halo_num, int):
        for i in halo_num:
            plt.axvline(x=initial_parameters.halo[i]['mass'].sum() * 0.701, color="k", ls="--")
    else:
        plt.axvline(x=initial_parameters.halo[halo_num]['mass'].sum()*0.701, color="k", ls="--")
    plt.legend(loc="best")
    plt.xscale("log")


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50)
    masses = w.smoothing_masses

    # Get predicted masses for particles in halos larger than halo[16320]

    ellipsoidal_predited_mass_0707 = get_predicted_analytic_mass(masses, ic, barrier="ST", cosmology="WMAP5",
                                                                 a=0.707, trajectories=None)
    ellipsoidal_predited_mass_075 = get_predicted_analytic_mass(masses, ic, barrier="ST", cosmology="WMAP5", a=0.75,
                                                                trajectories=None)
    ellipsoidal_predited_mass_1 = get_predicted_analytic_mass(masses, ic, barrier="ST", cosmology="WMAP5", a=1,
                                                              trajectories=None)
    spherical_prediced_mass = get_predicted_analytic_mass(masses, ic, barrier="spherical",
                                                          trajectories=None)

    true_mass = get_true_mass_particles_in_halos(ic, halo_min=None)

    print("ELLIPSOIDAL a = 1 - Number of trajectories which never cross threshold is "
          + str(len(np.where(np.isnan(ellipsoidal_predited_mass_1))[0])/len(true_mass)*100) + " %.")
    print("ELLIPSOIDAL a = 0.707 - Number of trajectories which never cross threshold is "
          + str(len(np.where(np.isnan(ellipsoidal_predited_mass_0707))[0])/len(true_mass)*100) + " %.")
    print("ELLIPSOIDAL a = 0.75 - Number of trajectories which never cross threshold is "
          + str(len(np.where(np.isnan(ellipsoidal_predited_mass_075))[0])/len(true_mass)*100) + " %.")
    print("SPHERICAL - Number of trajectories which never cross threshold is "
          + str(len(np.where(np.isnan(spherical_prediced_mass))[0])/len(true_mass)*100) + " %.")

    ###### Reproducing figure 2 in Sheth, Mo & Tormen paper (1999)  ######

    no_nan = np.where((~np.isnan(ellipsoidal_predited_mass_1)) & (~np.isnan(spherical_prediced_mass)) &
                      (~np.isnan(ellipsoidal_predited_mass_075)) & (~np.isnan(ellipsoidal_predited_mass_0707)))[0]

    random_particles = np.random.choice(no_nan, 10000)
    r = np.random.uniform(-0.2, 0.2, size=10000)

    scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_prediced_mass, ellipsoidal_predited_mass_1, true_mass,
                                                        random_particles, r)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/ellipsoidal/spherical_vs_ellips_a_1.pdf")
    plt.clf()

    scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_prediced_mass, ellipsoidal_predited_mass_075,
                                                        true_mass, random_particles, r)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/ellipsoidal/spherical_vs_ellips_a_075.pdf")
    plt.clf()

    scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_prediced_mass, ellipsoidal_predited_mass_0707,
                                                        true_mass, random_particles, r)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/ellipsoidal/spherical_vs_ellips_a_0707.pdf")
    plt.clf()

    ###### Reproducing figure 3 in Sheth, Mo & Tormen paper (1999)  ######

    # Take centre of mass of halos

    # halos = np.arange(0, 16321)
    # com_ids = np.zeros((len(halos)*10, ))
    # for i in range(len(halos)):
    #     com_ids[(i*10): ((i*10)+10)] = ic.halo[i]['iord'][:10]

    halos = np.arange(0, 16321)
    com_ids = np.zeros((len(halos), ))
    for i in range(len(halos)):
        com_ids[i] = ic.halo[i]['iord'][0]

    p_in_halos = get_particles_in_halos(ic)
    p_in_halos_no_nan = p_in_halos[no_nan]
    com_p_in_halos = p_in_halos_no_nan[np.in1d(p_in_halos_no_nan, com_ids)]
    ind_com = np.where(np.in1d(p_in_halos, com_p_in_halos))[0]

    r_com = np.random.uniform(-0.2, 0.2, size=len(com_p_in_halos))

    scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_prediced_mass, ellipsoidal_predited_mass_1, true_mass,
                                                        ind_com, r_com)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/ellipsoidal/COM_spherical_vs_ellips_a_1.pdf")
    plt.clf()

    scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_prediced_mass, ellipsoidal_predited_mass_075,
                                                        true_mass, ind_com, r_com)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/ellipsoidal/COM_spherical_vs_ellips_a_075.pdf")
    plt.clf()

    scatter_plot_spherical_ellipsoidal_predicted_masses(spherical_prediced_mass, ellipsoidal_predited_mass_0707,
                                                        true_mass, ind_com, r_com)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/ellipsoidal/COM_spherical_vs_ellips_a_0707.pdf")
    plt.clf()


    # Get distribution of predicted mass for halos `hal`

    pred_50 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k/pred_mass_50_smoothing.npy")
    pred_250 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k/pred_mass_250_smoothing.npy")
    pred_1500 = np.load("/Users/lls/Documents/CODE/stored_files/trajectories_sharp_k/pred_mass_1500_smoothing.npy")

    # also try with the predicted masses from sharp-k filter having modified k = a/boxsize

    hal = np.array([np.arange(810, 820), np.arange(0,10), np.arange(8000, 8010), np.arange(80,90)])
    for i in range(len(hal)):
        halos_i = hal[i]
        # mp.get_histogram_predicted_masses(hal, ic, particles_predicted=None, bins_num=20, normed=False,
        #                                predicted_new_filt=predicted_new_filt, predicted_250=predicted_250,
        #                                pred_1500=pred_1500)
        mp.get_histogram_predicted_masses_2(halos_i, ic, particles_predicted=None, bins_num=20, predicted_50=pred_50,
                                            predicted_250=pred_250, predicted_1500=pred_1500)

        plt.savefig("/Users/lls/Desktop/pred_mass_histograms/halos_" + str(halos_i.min()) + "_" + str(halos_i.max()) +
                    ".png")
        plt.clf()



