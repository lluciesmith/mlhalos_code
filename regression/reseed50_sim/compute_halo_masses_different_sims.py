import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
import pynbody


def get_simulations_halo_masses_each_particle(path_simulation, saving_path):
    initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_simulation + "IC.gadget3",
                                                            final_snapshot=path_simulation + "snapshot_099",
                                                            load_final=True, min_halo_number=0, max_halo_number=400,
                                                            min_mass_scale=3e10, max_mass_scale=1e15)
    # Get halo mass for each particle

    halo_id_particles = initial_params.final_snapshot['grp']
    halo_num = np.unique(halo_id_particles)[1:]

    halo_mass_particles = np.zeros(len(halo_id_particles))
    for i in halo_num:
        halo_mass_particles[np.where(halo_id_particles == i)[0]] = initial_params.halo[i]["mass"].sum()

    halo_mass_particles = pynbody.array.SimArray(halo_mass_particles)
    halo_mass_particles.units = initial_params.halo[0]["mass"].units
    np.save(saving_path + "halo_mass_particles.npy", halo_mass_particles)
    return halo_mass_particles


saving_path3 = "/share/data1/lls/standard_reseed3/"
path_simulation3 = "/share/hypatia/app/luisa/standard_reseed3/"
h3 = get_simulations_halo_masses_each_particle(path_simulation3, saving_path3)
del h3
del saving_path3
del path_simulation3

saving_path4 = "/share/data1/lls/standard_reseed4/"
path_simulation4 = "/share/hypatia/app/luisa/standard_reseed4/"
h4 = get_simulations_halo_masses_each_particle(path_simulation4, saving_path4)
del h4
del saving_path4
del path_simulation4

saving_path5 = "/share/data1/lls/standard_reseed5/"
path_simulation5 = "/share/hypatia/app/luisa/standard_reseed5/"
h5 = get_simulations_halo_masses_each_particle(path_simulation5, saving_path5)