import matplotlib
import numpy as np

matplotlib.use("macosx")
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
from mlhalos import window
from scripts.hmf import mass_predictions_ST as mp, hmf_tests
from scripts.ellipsoidal import hmf_tests

from mlhalos import parameters


def get_mass_comoving_new_filtering_scheme(boundary_schemes=0.038, bins_low=120):
    min_r = 0.0057291
    max_r = 0.2

    smoothing_radii_low = np.linspace(min_r, boundary_schemes, bins_low)
    smoothing_radii_high = []
    num = boundary_schemes
    while num < max_r:
        a = np.log10(num * 100) / 100
        smoothing_radii_high.append(num + a)
        num = num + a

    smoothing_radii = np.concatenate((smoothing_radii_low, smoothing_radii_high))
    smoothing_r_comoving = hmf_tests.radius_comoving_h_units(smoothing_radii, ic.initial_conditions)
    smoothing_m = hmf_tests.pynbody_r_to_m(smoothing_r_comoving, ic.initial_conditions)
    return smoothing_m



ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")
w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250, volume="sphere")

particles_in_halos = mp.get_particles_in_halos(ic, halo_min=None)

halos = [1600,1400, 100, 400, 800, 50]
num = np.array([ic.halo[i]['iord'][20] for i in halos])

bla = np.in1d(particles_in_halos, num)

# traj_new_filt = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/traj_new_filtering"
#                         ".npy")
traj_new_filt = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/traj_new_filtering_bound_030.npy")

traj_50 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/smoothed_density_contrast_50_particles_in_halos.npy")
#traj_smooth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_250_bins.npy")
traj_smooth_num = traj_50[bla, :]

del traj_50

traj = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sharp_k.npy")
traj_num = traj[num, :]

#traj_sphere = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sphere.npy")
#m_analy = mp.get_predicted_analytic_mass(w.smoothing_masses, ic, barrier="spherical", trajectories=traj)
#traj_sphere_num = traj_sphere[num, :]
#del traj_sphere
del traj

#w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250, volume="sphere")
#w1 = window.WindowParameters(initial_parameters=ic, num_filtering_scales=250, volume="sharp-k")
#m_analy = mp.get_predicted_analytic_mass(w.smoothing_masses, ic, barrier="spherical", trajectories=traj)

particles_in_halos = mp.get_particles_in_halos(ic, halo_min=None)

smoothing_radii_50 = np.linspace(0.0057291, 0.20, 50)
smoothing_r_comoving_50 = hmf_tests.radius_comoving_h_units(smoothing_radii_50, ic.initial_conditions)
smoothing_m_50 = hmf_tests.pynbody_r_to_m(smoothing_r_comoving_50, ic.initial_conditions)

smoothing_radii_250 = np.linspace(0.0057291, 0.20, 250)
smoothing_r_comoving_250 = hmf_tests.radius_comoving_h_units(smoothing_radii_250, ic.initial_conditions)
smoothing_m_250 = hmf_tests.pynbody_r_to_m(smoothing_r_comoving_250, ic.initial_conditions)

m_new_filt = get_mass_comoving_new_filtering_scheme()

for ran in range(len(num)):
    plt.clf()
    #a = np.where(particles_in_halos == num[ran])

    plt.figure(figsize=(8,6))
    ok = np.where(traj_smooth_num[ran] != 0)[0]
    plt.plot(smoothing_m_250, traj_num[ran], label="ID " + str(num[ran]))
    plt.plot(smoothing_m_250, traj_num[ran], label="ID " + str(num[ran]))
    plt.plot(smoothing_m_50[ok], traj_smooth_num[ran, ok], color="g")

    #ok = np.where(traj_sphere_num[ran] != 0)[0]
    #plt.plot(w1.smoothing_masses[ok], traj_sphere_num[ran, ok], label="mass sphere", color="grey")

    plt.axhline(y=1.01686, color="k", ls="--")

    #plt.axvline(x=m_analy[a], color="r", ls="--", label="predicted")
    true_mass = ic.halo[halos[ran]]['mass'].sum()
    plt.axvline(x=true_mass, color="grey", ls="--", label="true")


    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"$\mathrm{M}/\mathrm{M}_{\odot}$")
    plt.ylabel(r"$\delta + 1$")
    plt.tight_layout()
    plt.savefig("/Users/lls/Documents/CODE/stored_files/trajectories_upcrossing/trajectory_id_" + str(num[ran]) +
                ".png")
    #plt.show()

def plot_trajectories_different_radii(ic, num):
    particles_in_halos = mp.get_particles_in_halos(ic, halo_min=None)

    traj_50 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k/smoothed_density_contrast_50_particles_in_halos.npy")
    traj_250 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_sharp_k"
                       "/smoothed_density_contrast_250_particles_in_halos.npy")

    traj_smooth = np.load("/Users/lls/Documents/CODE/stored_files/hmf/trajectories_250_bins.npy")
    traj_smooth = traj_smooth[particles_in_halos, :]

    traj_sphere = np.load("/Users/lls/Documents/CODE/stored_files/hmf/250_traj_all_sharp_k_filter_volume_sphere.npy")
    traj_sphere_particles = traj_sphere[particles_in_halos, :]

    some_p = np.random.choice(len(particles_in_halos), 100)

    #traj_50_p = traj_50[some_p[num], :]
    traj_250_p = traj_250[some_p[num], :]
    traj_sphere_p = traj_sphere_particles[some_p[num], :]

    smoothing_radii_50 = np.linspace(0.0057291, 0.20, 50)
    smoothing_r_comoving_50 = hmf_tests.radius_comoving_h_units(smoothing_radii_50, ic.initial_conditions)
    smoothing_m_50 = hmf_tests.pynbody_r_to_m(smoothing_r_comoving_50, ic.initial_conditions)

    smoothing_radii_250 = np.linspace(0.0057291, 0.20, 250)
    smoothing_r_comoving_250 = hmf_tests.radius_comoving_h_units(smoothing_radii_250, ic.initial_conditions)
    smoothing_m_250 = hmf_tests.pynbody_r_to_m(smoothing_r_comoving_250, ic.initial_conditions)


    traj_250_p = traj_250[some_p[num], :]
    #traj_sphere_p = traj_sphere_particles[some_p[num], :]
    #traj_50_p = traj_50[some_p[num], :]
    traj_new_filt_p = traj_new_filt[some_p[num], :]

    plt.figure(figsize=(8, 6))
    ok_sph = np.where(traj_sphere_p != 0)[0]
    ok_250 = np.where(traj_250_p != 0)[0]
    ok_50 = np.where(traj_50_p != 0)[0]
    ok_new = np.where(traj_new_filt_p != 0)[0]
    #plt.plot(smoothing_m_250, traj_250_p)
    #plt.plot(smoothing_m_50[ok_50], traj_50_p[ok_50], label="50 evenly R")
    plt.plot(smoothing_m[ok_new], traj_new_filt_p[ok_new], label="new filt", color="b")
    plt.scatter(smoothing_m[ok_new], traj_new_filt_p[ok_new],  color="b")
    #plt.plot(w.smoothing_masses[ok_sph] * 0.701, traj_sphere_p[ok_sph], label="250 evenly log(M)")
    plt.plot(smoothing_m_250[ok_250], traj_250_p[ok_250], label="250 evenly R", color="g")
    plt.scatter(smoothing_m_250[ok_250], traj_250_p[ok_250], color="g")
    plt.axhline(y=1.01686, color="k", ls="--")
    #plt.axvline(x=m, color="r", ls="--", label="predicted")
    plt.legend(loc="best")
    plt.xscale("log")
    plt.xlabel(r"$\mathrm{M}/\mathrm{M}_{\odot}$")
    plt.ylabel(r"$\delta + 1$")

    plt.figure(figsize=(8, 6))
    plt.plot(smoothing_m[(smoothing_m<10**12)], traj_new_filt_p[(smoothing_m<10**12)], label="new filt", color="b")
    plt.scatter(smoothing_m[(smoothing_m<10**12)], traj_new_filt_p[(smoothing_m<10**12)], color="b")
    plt.plot(smoothing_m_250[(smoothing_m_250<10**12)], traj_250_p[(smoothing_m_250<10**12)], label="250 evenly R",
             color="g")
    plt.axhline(y=1.01686, color="k", ls="--")
    plt.scatter(smoothing_m_250[(smoothing_m_250<10**12)], traj_250_p[(smoothing_m_250<10**12)], color="g")
    plt.legend(loc="best")
    plt.xlabel(r"$\mathrm{M}/\mathrm{M}_{\odot}$")
    plt.ylabel(r"$\delta + 1$")
    plt.xlim(0, 10**12)

