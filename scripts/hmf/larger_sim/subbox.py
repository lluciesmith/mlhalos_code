import sys
# sys.path.append("/Users/lls/Documents/mlhalos_code")
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from multiprocessing import Pool

# initial_snapshot = "/Users/lls/Documents/CODE/sim200/sim200.gadget3"
# ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot, path="/Users/lls/Documents/CODE/")
#
# sim = ic_200.initial_conditions
# assert sim['pos'].units == "kpc"
# rho = sim['rho']
# assert rho.units == "Msol kpc**-3"
#
# rhoM = pynbody.analysis.cosmology.rho_M(ic_200.initial_conditions, unit="Msol kpc**-3")
# rhoM2 = len(sim) * sim['mass'][0] / (200 * 0.01 / 0.701 * 10**3)**3



def range1(coordinate):
    boxsize = 200
    return (coordinate >=0) & (coordinate < boxsize/4)

def range2(coordinate):
    boxsize = 200
    return (coordinate >= boxsize/4) & (coordinate < boxsize/2)

def range3(coordinate):
    boxsize = 200
    return (coordinate >= boxsize/2) & (coordinate < 3*boxsize/4)

def range4(coordinate):
    boxsize = 200
    return (coordinate >= 3*boxsize/4) & (coordinate < boxsize)


def get_subboxes_rms(snapshot, subboxes=64):
    x = snapshot['x'].in_units("Mpc a h^-1")
    y = snapshot['y'].in_units("Mpc a h^-1")
    z = snapshot['z'].in_units("Mpc a h^-1")

    ids = np.array([np.where(a(x) & b(y) & c(z))[0]
                    for a in [range1, range2, range3, range4]
                    for b in [range1, range2, range3, range4]
                    for c in [range1, range2, range3, range4]])

    rho = snapshot['rho']

    mean_densities = np.zeros((subboxes,))
    for i in range(subboxes):
        #t = pynbody.analysis.halo.center(sim[ids[i]], vel=False)
        #sim.wrap()
        mean_densities[i] = np.mean(rho[ids[i]])

    rhoM = pynbody.analysis.cosmology.rho_M(snapshot, unit="Msol kpc**-3")
    delta = mean_densities/rhoM - 1
    rms = np.sqrt(np.mean(delta ** 2))
    return mean_densities, rms


# Cubic subboxes

def get_cuboid(sim, l):
    ids_cuboid = np.where((sim["x"] > -l / 2) & (sim["x"] < l / 2) & (sim["y"] > -l / 2) & (sim["y"] < l / 2)
                          & (sim["z"] > -l / 2) & (sim["z"] < l / 2))[0]
    return ids_cuboid


def get_ids_subbox_centered_on_particle(sim, particle_id, length_subbox):
    pynbody.analysis.halo.center(sim, mode="ind", ind=particle_id, vel=False)
    sim.wrap()

    ids_cuboid = get_cuboid(sim, length_subbox)
    return ids_cuboid


def get_mean_density_cubic_subbox_randomly_centered(sim, realisations=100):
    n = np.random.choice(range(512 ** 3), realisations)
    den_mean = np.empty((realisations,))

    l = 50 * 0.01 / 0.701 * 10 ** 3
    rho = sim['rho']

    for i in range(len(n)):
        ids_cuboid = get_ids_subbox_centered_on_particle(sim, n[i], l)
        den = rho[ids_cuboid]
        den_mean[i] = np.mean(den)
    return den_mean


def get_rms_subboxes_randomly_centered(sim, realisations=100):
    den_mean = get_mean_density_cubic_subbox_randomly_centered(sim, realisations=realisations)

    rhoM = pynbody.analysis.cosmology.rho_M(sim, unit="Msol kpc**-3")
    delta = den_mean / rhoM - 1
    rms = np.sqrt(np.mean(delta ** 2))
    return den_mean, rms



# Spheres

def get_mean_density_spheres_randomly_centred(sim, radius, realisations=100):
    n = np.random.choice(range(512 ** 3), realisations)
    den_mean = np.empty((realisations,))

    for i in range(len(n)):
        pynbody.analysis.halo.center(sim, mode="ind", ind=n[i], vel=False)
        sim.wrap()
        den = sim[pynbody.filt.Sphere(radius)]['rho']
        den_mean[i] = np.mean(den)
    return den_mean


def get_variance_spheres_randomly_centred(sim, radius, realisations=100):
    den_mean = get_mean_density_spheres_randomly_centred(sim, radius, realisations=realisations)

    rhoM = pynbody.analysis.cosmology.rho_M(sim, unit="Msol kpc**-3")
    delta = den_mean / rhoM - 1
    rms = np.sqrt(np.mean(delta ** 2))
    return den_mean, rms

def plot_delta(ic):
    f = ic.initial_conditions
    pynbody.analysis.halo.center(f, mode="ind", ind=i, vel=False)
    f.wrap()
    rho_mean = f['mass'].sum()/(200*10**3 * 0.01 / 0.701)**3
    rho_mean.units = "Msol kpc**-3"
    f['delta'] = (f['rho'] - rho_mean) / rho_mean
    pynbody.plot.sph.image(ic.initial_conditions, qty="delta", width=ic.initial_conditions.properties['boxsize'],
                        log=False)


if __name__ == "__main__":
    #initial_snapshot = "/home/app/scratch/sim200.gadget3"
    # initial_snapshot = "/Users/lls/Documents/CODE/standard200/standard200.gadget3"
    # final_snapshot = "/Users/lls/Documents/CODE/standard200/snapshot_011"
    #ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot
                                                    #final_snapshot=final_snapshot,
                                                    #path="/Users/lls/Documents/CODE/"
    #                                                )
    ic_50 = parameters.InitialConditionsParameters(
        #path="/Users/lls/Documents/CODE/"
        )

    #mean_densities_64, rms_64 = get_subboxes_rms(ic.initial_conditions, subboxes=64)
    #mean_den_random, rms_random = get_rms_subboxes_randomly_centered(ic.initial_conditions)

    # rho_200 = ic_200.initial_conditions['rho']
    # np.save("/home/lls/stored_files/mean_densities/rho_200.npy", rho_200)
    #
    # rho_50 = ic_50.initial_conditions['rho']
    # np.save("/home/lls/stored_files/mean_densities/rho_50.npy", rho_50)

    #rho_200 = np.load("/home/lls/stored_files/mean_densities/rho_200.npy")
    rho_50 = np.load("/home/lls/stored_files/mean_densities/rho_50.npy")


    # def center_and_mean_density_sphere_L200(particle_id):
    #     sim = ic_200.initial_conditions
    #     sim['rho'] = rho_200
    #     l = 50 * 0.01 / 0.701 * 10 ** 3
    #
    #     pynbody.analysis.halo.center(sim, mode="ind", ind=particle_id, vel=False)
    #     sim.wrap()
    #
    #     den = sim[pynbody.filt.Sphere("8 Mpc h^-1 a")]['rho']
    #     return np.mean(den)

    def center_and_mean_density_sphere_L50(particle_id):
        sim = ic_50.initial_conditions
        sim['rho'] = rho_50
        l = 50 * 0.01 / 0.701 * 10 ** 3

        pynbody.analysis.halo.center(sim, mode="ind", ind=particle_id, vel=False)
        sim.wrap()

        den = sim[pynbody.filt.Sphere("8 Mpc h^-1 a")]['rho']
        return np.mean(den)

    # def center_and_mean_density_cube(particle_id):
    #     sim = ic_200.initial_conditions
    #     l = 50 * 0.01 / 0.701 * 10 ** 3
    #
    #     pynbody.analysis.halo.center(sim, mode="ind", ind=particle_id, vel=False)
    #     sim.wrap()
    #
    #     ids_cuboid = np.where((sim["x"] > -l / 2) & (sim["x"] < l / 2) & (sim["y"] > -l / 2) & (sim["y"] < l / 2)
    #                           & (sim["z"] > -l / 2) & (sim["z"] < l / 2))[0]
    #     den = rho_200[ids_cuboid]
    #     return np.mean(den)


    # pool = Pool(processes=24)
    # f = center_and_mean_density_cube
    # n = np.random.choice(range(512 ** 3), 500)
    # mean_densities = pool.map(f, n)
    # pool.close()
    # pool.join()
    # #
    # # # mean_den_subboxes = np.zeros((50, 100))
    # # # for i in range(50):
    # # #     mean_den_subboxes[i] = get_mean_density_cubic_subbox_randomly_centered(ic_200.initial_conditions,
    # # #                                                                            realisations=100)
    # np.save("/home/lls/stored_files/mean_densities/mean_den_subboxes_pool_3.npy", mean_densities)


    # np.save("/Users/lls/Desktop/mean_densities/mean_den_subboxes.npy", mean_den_subboxes)
    # del mean_den_subboxes
    #
    # mean_den_spheres = np.zeros((50, 100))
    # for i in range(50):
    #     mean_den_spheres[i] = get_mean_density_spheres_randomly_centred(ic_200.initial_conditions, "8 Mpc h^-1 a",
    #                                                                     realisations=100)
    # np.save("/home/lls/stored_files/mean_densities/mean_den_spheres_L200.npy", mean_den_spheres)
    # del mean_den_spheres
    #
    # mean_den_spheres_L50 = np.zeros((50, 100))
    # for i in range(50):
    #     mean_den_spheres_L50[i] = get_mean_density_spheres_randomly_centred(ic_50.initial_conditions, "8 Mpc h^-1 a",
    #                                                                         realisations=100)
    # np.save("/home/lls/stored_files/mean_densities/mean_den_spheres_L50.npy", mean_den_spheres_L50)
    # del mean_den_spheres_L50

    pool = Pool(processes=24)
    f = center_and_mean_density_sphere_L50
    n = np.random.choice(range(256 ** 3), 500)
    mean_densities_sphere = pool.map(f, n)
    pool.close()
    pool.join()

    np.save("/home/lls/stored_files/mean_densities/mean_den_spheres_L50_2.npy", mean_densities_sphere)


    # Tomorrow's stuff

    # rhoM = pynbody.array.SimArray([pynbody.analysis.cosmology.rho_M(ic_50.initial_conditions, unit="Msol "
    #                                                                                                "kpc**-3")])
    # rhoM.units = "Msol kpc**-3"
    #
    # rho_mean = ic_50.initial_conditions['mass'].sum() / ic.initial_conditions.properties['boxsize']**3
    #
    # rho = pynbody.array.SimArray([10**8])
    # rho.units = "Msol kpc**-3"

