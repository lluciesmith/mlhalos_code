import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import pynbody
import numpy as np
from mlhalos import parameters
from mlhalos import distinct_colours


path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"
ids_tested = np.load(path + "tested_ids.npy")

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
f = ic.final_snapshot
f.physical_units("kpc")
h = ic.halo


Fps_ids_den = np.load("/Users/lls/Desktop/Fps_th_28_den.npy")
Fps_ids_den_shear = np.load("/Users/lls/Desktop/Fps_th_28_den_den_sub_ell.npy")

halos = [0, 10, 100, 200, 300]
c = distinct_colours.get_distinct(len(halos))

overden_all = np.zeros((len(halos), 11))
n_tot_all = np.zeros((len(halos), 11))
ind = np.zeros((len(halos), 11))

for j in range(len(halos)):
    halo_number = halos[j]
    col = c[j]
    overden_first = 400

    pynbody.analysis.halo.center(h[halo_number], vel=False)
    f.wrap()

    r_initial = pynbody.analysis.halo.virial_radius(h[halo_number], overden=overden_first)
    overden_initial = overden_first

    r_FPs_den = f[Fps_ids_den]['r']
    r_FPs_den_ell = f[Fps_ids_den_shear]['r']

    n = len(np.where(r_FPs_den <= r_initial)[0])

    # new
    r_all = f[ids_tested]['r']
    n_all_len = len(np.where(r_all <= r_initial)[0])

    overden = []
    overden.append(overden_initial)

    n_tot = []
    # n_tot.append(n)
    # new
    n_tot.append(n/n_all_len)

    den_mean = f.properties["omegaM0"] * pynbody.analysis.cosmology.rho_crit(f, z=0)

    for i in range(10):
        r_2 = r_initial * 2
        mass = f[pynbody.filt.Sphere(r_2)]['mass'].sum()
        density = mass / (4/3*np.pi*(r_2**3))

        overden_2 = density/den_mean
        # d = overden_2 - overden_initial
        # overden.append(float(d))
        overden.append(float(overden_2))

        # num_FPs = len(np.where((r_FPs_den > r_initial) & (r_FPs_den <= r_2))[0])
        num_FPs = len(np.where(r_FPs_den <= r_2)[0])
        # n_tot.append(num_FPs)

        # new
        n_all = len(np.where(r_all <= r_2)[0])
        n_tot.append(num_FPs/n_all)

        r_initial = r_2
        overden_initial = overden_2


    n_tot_all[j] = np.array(n_tot)
    overden_all[j] = np.array(overden)
    ind[j] = np.argsort(overden)[::-1]

    # plt.plot(overden[ind], n_tot[ind]/np.sum(n_tot), color=col)
for i in range(len(halos)):
    ind = np.argsort(overden_all[i])[::-1]
    plt.plot(overden_all[i][ind], n_tot_all[i][ind]/np.sum(n_tot_all[i][ind]), color=c[i], label="halo " + str(
        halos[i]))
    plt.scatter(overden_all[i][ind], n_tot_all[i][ind]/np.sum(n_tot_all[i][ind]), color=c[i], s=20)

# n, b = np.histogram(overden_all[0], 10)
#
# for k in range(len(b) - 1):
#
#     for i in range(len(b) - 1):
#         ind = (overden_all >= b[i]) & (overden_all < b[i + 1])
#         print(overden_all[ind])
#         ov[i] = np.median(overden_all[ind])
#
#
# medians = np.median(n_tot_all[overden_all[:,]])
#
# plt.scatter(overden[ind], n_tot[ind]/np.sum(n_tot), color=col, s=20, label="halo " + str(halo_number))

plt.legend(loc="best")
plt.xlabel(r"$\rho / \rho_M$")
plt.ylabel(r"$\mathrm{N_{FPs}} / \sum{\mathrm{N_{FPs}}}$")
plt.xscale("symlog")
