import numpy as np
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
from mlhalos import parameters
from utils import radius_func


ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
f = ic.final_snapshot
h = ic.halo
path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"


y_true = np.load(path + "density_only/true_den.npy")
ids_tested = np.load(path + "tested_ids.npy")
y_true_modified = np.copy(y_true)

massive_halos = np.arange(0, 11)
# ids_in_massive_halos = ids_tested[(f[ids_tested]['grp'] >= massive_halos[0]) &
#                                   (f[ids_tested]['grp'] <= massive_halos[-1])]

for i in massive_halos:
    vir_i = radius_func.virial_radius(i, f=f, h=h, overden=200, particles="all")
    r_particles = radius_func.radius_particle(ids_tested, halo_ID=i, f=f, h=h, center=False)
    assert vir_i.units == r_particles.units

    index_in_virial_region = np.where(r_particles <= vir_i)[0]
    # ids_virial_particles = ids_tested[index_in_virial_region]

    # index = np.in1d(ids_tested, ids_virial_particles)
    y_true_modified[index_in_virial_region] = 1

np.save("/Users/lls/Desktop/modified_true_labels.npy", y_true_modified)
