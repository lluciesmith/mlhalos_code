import numpy as np
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from utils import classification_results
from utils import radius_func as rad


def find_indices_of_particle_ids_in_halo(all_halos_particles, halo_ID):
    pos = np.in1d(all_halos_particles, halo_ID)
    return pos


def get_radius_particles_of_halo(all_radii_particles, all_halos_particles, halo_ID, pos=None):
    if pos is None:
        pos = find_indices_of_particle_ids_in_halo(all_halos_particles, halo_ID)
    r = all_radii_particles[pos]
    return r


def get_fraction_virial_rad_in_halo(all_radii_particles, all_halos_particles, halo_ID, pos=None):
    """

    Args:
        halo_ID: The halo you are interested in
        all_halos_particles: Array of halo-ID each particle in ids_in
        radii_in_particles:

    Returns:

    """
    virial_radius = rad.virial_radius(halo_ID)
    radius_particles = get_radius_particles_of_halo(all_radii_particles, all_halos_particles, halo_ID, pos=pos)
    return radius_particles/virial_radius


def fraction_virial_radius(all_particles, all_radii_particles, all_halos_particles):
    radius_fraction = np.zeros((len(all_particles),))
    set_halos = set(all_halos_particles)

    for halo_ID in set_halos:
        pos = find_indices_of_particle_ids_in_halo(all_halos_particles, halo_ID)
        if halo_ID == 336:
            radius_fraction[pos] == np.NaN
        else:
            r_fraction = get_fraction_virial_rad_in_halo(all_radii_particles, all_halos_particles, halo_ID, pos=pos)
            radius_fraction[pos] = r_fraction
    return radius_fraction


##################### SCRIPT ########################

if __name__ == "__main__":
    f, h = classification_results.load_final_snapshot_and_halos()
    old_radius_properties = np.load('/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_in_ids'
                                    '.npy')

    ids_in = old_radius_properties[:, 0].astype('int')
    radius_in = old_radius_properties[:, 1]
    halos_ids = f[ids_in]['grp']

    radius_fraction_in = fraction_virial_radius(ids_in, radius_in, halos_ids)

    np.save('/Users/lls/Documents/CODE/stored_files/all_out/radii_files/fraction_radii_in_new.npy',
            radius_fraction_in)