import sys
sys.path.append("/home/lls/mlhalos_code")
#sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody
from multiprocessing import Pool

ics = "/share/data1/lls/sim200/simulation/standard200.gadget3"
f = "/share/data1/lls/sim200/simulation/snapshot_011"
# ics = "/Users/lls/Documents/mlhalos_files/standard200/standard200.gadget3"
# f = "/Users/lls/Documents/mlhalos_files/standard200/snapshot_011"
ic = parameters.InitialConditionsParameters(initial_snapshot=ics, final_snapshot=f)
final_snap = ic.final_snapshot
h = ic.halo
halo_ids = np.arange(len(ic.halo))

h_id_higher_mass = 22000
h_id_lower_mass = 43500 # mass 1.00171 x 10^12 Msol
halos_in_mass_range = np.arange(h_id_higher_mass, h_id_lower_mass)

# Calculate the density within 10 Mpc of each within the mass range above

radius_str = "10 Mpc"
V = pynbody.array.SimArray(4/3 * np.pi * 10**3)
V.units = "Mpc**3"


def get_rho(number):
    halo_ID = halos_in_mass_range[number]
    pynbody.analysis.halo.center(final_snap[h[halo_ID].properties['mostboundID']], vel=False)
    final_snap.wrap()
    pynbody.analysis.halo.center(h[halo_ID], vel=False)
    sphere = final_snap.dm[pynbody.filt.Sphere(radius_str)]
    m = sphere['mass'].sum()
    rho_ID = m/V
    return rho_ID


pool = Pool(processes=60)
ids = np.arange(len(halos_in_mass_range))
rho_halos = pool.map(get_rho, ids)
pool.close()
pool.join()

rho_halos = np.array(rho_halos)
np.save("/share/data1/lls/sim200/rho_environment.npy", rho_halos)

# rho_halos = np.zeros((len(halos_in_mass_range)))
# for i in range(len(halos_in_mass_range)):
#     halo_ID = halos_in_mass_range[i]
#     pynbody.analysis.halo.center(final_snap[h[halo_ID].properties['mostboundID']], vel=False)
#     final_snap.wrap()
#     pynbody.analysis.halo.center(h[halo_ID], vel=False)
#     sphere = final_snap.dm[pynbody.filt.Sphere(radius_str)]
#     m = sphere['mass'].sum()
#     rho_halos[i] = m/V
#     print("Done halo " + str(halo_ID))
#
# #np.save("/Users/lls/Documents/mlhalos_files/standard200/rho_environment.npy", rho_halos)
# np.save("/share/data1/lls/sim200/rho_environment.npy", rho_halos)