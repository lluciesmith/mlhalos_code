import numpy as np
import sys
sys.path.append("/home/lls/mlhalos_code/")
import numpy as np
from mlhalos import parameters
from mlhalos import inertia
from multiprocessing import Pool

path = "/share/data2/lls/regression/local_inertia/tensor/"
ran_5k = np.load("/share/data2/lls/regression/local_inertia/tensor/ran_5k.npy")


def pool_local_inertia(particle_id):
    li, eigi = In.get_local_inertia_single_id(particle_id, snapshot, r_smoothing, rho)
    np.save(path + "random/inertia_tensor_particle_" + str(particle_id) + ".npy", li)
    np.save(path + "random/eigenvalues_particle_" + str(particle_id) + ".npy", eigi)

    # print("Done and saved particle " + str(particle_id))
    return li

ic = parameters.InitialConditionsParameters(load_final=True)
In = inertia.LocalInertia(ran_5k, initial_parameters=ic)

snapshot = ic.initial_conditions
rho = snapshot["rho"]
filtering_scales = In.filt_scales
r_smoothing = In.filter_parameters.smoothing_radii.in_units(snapshot["pos"].units)[filtering_scales]

pool = Pool(processes=40)
li_particles = pool.map(pool_local_inertia, ran_5k)
pool.close()
pool.join()
