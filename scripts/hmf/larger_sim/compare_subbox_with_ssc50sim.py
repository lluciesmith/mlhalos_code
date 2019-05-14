import numpy as np
from mlhalos import parameters
import pynbody

initial_snapshot = "/Users/lls/Documents/CODE/standard200/standard200.gadget3"
final_snapshot = "/Users/lls/Documents/CODE/standard200/snapshot_011"
ic_200 = parameters.InitialConditionsParameters(initial_snapshot=initial_snapshot,
                                                final_snapshot=final_snapshot, path="/Users/lls/Documents/CODE/")

sim = ic_200.initial_conditions
assert sim['pos'].units == "kpc"
rho = sim['rho']
assert rho.units == "Msol kpc**-3"
rho_mean = np.mean(rho)

id = 13527457
t = pynbody.analysis.halo.center(sim[id], vel=False)
sim.wrap()

l = 713
ids_cuboid = np.where((sim["x"] > -l / 2) & (sim["x"] < l / 2) & (sim["y"] > -l / 2) & (sim["y"] < l / 2)
                      & (sim["z"] > -l / 2) & (sim["z"] < l / 2))[0]
den_cuboid = rho[ids_cuboid]
delta_cuboid = np.mean(den_cuboid)/rho_mean - 1
# Find ssc in loaded files with similar mean density fluctuation

