"""

Compute the radius of each "in" particle w.r.t. the "shrinking sphere" centre of the halo the particle belongs to.

The output array ``radii_ids_in`` [n_radii,] is an array such that each element is the radius of the corresponding
particle in the
array ``all_ids_in``.
"""

import numpy as np
import pynbody
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from utils import classification_results


def find_radius_particles_centred_in_halo(particle_ids, halo_ID, f=None, h=None):
    if f is None and h is None:
        f, h = classification_results.load_final_snapshot_and_halos()

    # centre the snapshot on the "shrinking sphere" radius
    pynbody.analysis.halo.center(f[h[halo_ID].properties['mostboundID']], vel=False)
    f.wrap()
    pynbody.analysis.halo.center(h[halo_ID], vel=False)
    radii = np.array([f[particle]['r'] for particle in particle_ids])
    return radii


def get_radius_of_particles(particles, f=None, h=None):
    if f is None and h is None:
        f, h = classification_results.load_final_snapshot_and_halos()

    radii_particles = np.zeros((len(particles),))
    halos = f[particles]['grp']

    for halo_id in set(halos):
        index = [halos == halo_id]
        particles_halo_id = particles[index]
        radii_particles[index] = find_radius_particles_centred_in_halo(particles_halo_id, halo_id, f=f, h=h)
    return radii_particles


if __name__ == "__main__":
    ids_in = np.load('/Users/lls/Documents/CODE/stored_files/all_out/all_ids_in.npy').astype('int')
    radii_ids_in = get_radius_of_particles(ids_in)
    np.save('/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_ids_in_check.npy', radii_ids_in)

