import sys
sys.path.append("/home/lls/mlhalos_code")
#sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from mlhalos import parameters


ics = "/share/data1/lls/sim200/simulation/standard200.gadget3"
f = "/share/data1/lls/sim200/simulation/snapshot_011"

ic = parameters.InitialConditionsParameters(initial_snapshot=ics, final_snapshot=f)

final_snap = ic.final_snapshot
h = ic.halo
halo_mass = np.zeros((len(ic.final_snapshot),))

halo_ids = np.arange(len(ic.halo))
for halo_id in range(len(halo_ids)):
    mass_halo = ic.halo[halo_id]['mass'].sum()
    halo_mass[ic.halo[halo_id]['iord']] = mass_halo

np.save("/share/data1/lls/sim200/halo_mass_particles.npy", halo_mass)