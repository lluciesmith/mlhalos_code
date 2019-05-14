import numpy as np
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from utils import radius_func
from utils import classification_results as res
import pynbody
from multiprocessing import Pool


def get_particles_in_virial_region_not_in_halo(halo_id, f=None, h=None):
    if f is None and h is None:
        f, h = res.load_final_snapshot_and_halos()

    r_only_FOF = radius_func.virial_radius(halo_id, cen="halo ssc", f=f, h=h, overden=200, particles="FOF")
    particles_FOF = h[halo_id][pynbody.filt.Sphere(r_only_FOF)]['iord']

    r_all = radius_func.virial_radius(halo_id, cen="halo ssc", f=f, h=h, overden=200, particles="all")
    particles_all = f[pynbody.filt.Sphere(r_all)]['iord']

    diff_particles = np.asarray(np.setdiff1d(particles_all, particles_FOF))

    # ignore particles in halo[halo-id]. Take only particles inside region with r=r_all which do not belong to the halo.
    # Does the algorithm correctly classify them as out? What fraction of those are false positives, i.e. particle
    # which we label as "out" but that the algorithm sees as "in"?

    diff_negative_particles = diff_particles[f[diff_particles]['grp'] != halo_id]
    return diff_negative_particles


def get_percentage_category_from_subset_particles(subset_particles, category="false positives"):
    ids, pred, true = res.load_classification_results()

    if category == "false positives":
        category_particles = res.get_false_positives(ids, pred, true)

    elif category == "true negatives":
        category_particles = res.get_true_negatives(ids, pred, true)

    elif category == "false negatives":
        category_particles = res.get_false_negatives(ids, pred, true)

    elif category == "true positives":
        category_particles = res.get_true_positives(ids, pred, true)

    else:
        category_particles = category

    intersection = np.intersect1d(category_particles, subset_particles)
    percentage = len(intersection) / len(subset_particles) * 100
    return percentage


if __name__ == "__main__":

    halos = np.arange(0, 401)

    # WARNING - IGNORE HALO 336
    not_FOF_particles_virial_region = [np.asarray(get_particles_in_virial_region_not_in_halo(halo)) if halo != 336
                                       else 0 for halo in halos]
    not_FOF_particles_virial_region = np.hstack(not_FOF_particles_virial_region)
    # np.save('/Users/lls/Documents/CODE/stored_files/all_out/not_FOF_particles_virial_region.npy',
    # not_FOF_particles_virial_region)

    # What fraction of those are false positives, i.e. particles
    # which we label as "out" but that the algorithm sees as "in"?

    num_FPs_in_virial_region = get_percentage_category_from_subset_particles(not_FOF_particles_virial_region,
                                                                             category="false positives")
    num_TNs_in_virial_region = get_percentage_category_from_subset_particles(not_FOF_particles_virial_region,
                                                                             category="true negatives")

