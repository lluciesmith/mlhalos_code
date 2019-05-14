import numpy as np
import pynbody
import sys

from matplotlib import pyplot as plt

from utils.plot import get_log_spaced_bins_flat_distribution

sys.path.append('/Users/lls/Documents/CODE/git/mlhalos_code/scripts')
from utils import classification_results as res, mass
from scripts.EPS import EPS_predictions as EPS_pred


############# FUNCTIONS TO FIND RADIUS OF PARTICLES AND VIRIAL RADIUS OF HALOS #############

def virial_radius(halo_id, cen="halo ssc", f=None, h=None, overden=200, particles="FOF"):
    """ Return virial radius of the halo[halo_id] (in kpc)"""

    if f is None and h is None:
        f, h = res.load_final_snapshot_and_halos()

    if cen == "most bound":
        pynbody.analysis.halo.center(f[h[halo_id].properties['mostboundID']], vel=False)
        f.wrap()
    elif cen == "halo ssc":
        pynbody.analysis.halo.center(f[h[halo_id].properties['mostboundID']], vel=False)
        f.wrap()
        pynbody.analysis.halo.center(h[halo_id], vel=False)
    else:
        NameError("Invalid centering")

    if particles == "FOF":
        return pynbody.analysis.halo.virial_radius(h[halo_id], overden=overden)
    elif particles == "all":
        return pynbody.analysis.halo.virial_radius(f, overden=overden)
    else:
        NameError("Choose if to include in the virial region only FOF particles of the halo, or all particles.")

    print("Done halo " + str(halo_id))


def radius_particle(particle_ID, halo_ID=None, f=None, h=None, center=True):
    """Return radius of particle inside halo it belongs to"""
    if f is None and h is None:
        f, h = res.load_final_snapshot_and_halos()

    if halo_ID is None:
        halo_ID = f['grp'][particle_ID]

    # Need to centre on the particle's halo before finding its radius.
    # center on most bound ID and wrap so you avoid issues with halos at boundaries of box
    if center is True:
        pynbody.analysis.halo.center(f[h[halo_ID].properties['mostboundID']], vel=False)
        f.wrap()
        pynbody.analysis.halo.center(h[halo_ID], vel=False)

    r = f[particle_ID]['r']
    return r


def get_a_fraction_virial_radius(halo_ID, fraction, f=None, h=None):
    """Return value of r = fraction * virial radius of halo_ID"""
    r = virial_radius(halo_ID, f=f, h=h) * fraction
    if r < 25.6:
        print("Warning: The radius chosen is smaller than the simulation's softening scale (25.6 kpc)")
    return str(r) + " kpc"


def get_ratio_particle_radius_and_virial_radius_halo(particle_ID, halo_ID=None, f=None, h=None):

    if f is None and h is None:
        f, h = res.load_final_snapshot_and_halos()

    if halo_ID is None:
        halo_ID = f[particle_ID]['grp'][0]

    r = radius_particle(particle_ID, halo_ID=halo_ID, f=f, h=h)
    virial_r = virial_radius(halo_ID, f=f, h=h)

    ratio = r / virial_r
    return ratio[0]


############### SPLIT PARTICLES IN RADIUS BINS ###############


def get_particle_ids_inner_fraction_virial_halo(halo_IDs, fraction, f=None, h=None, percentage_min=None):

    if f is None and h is None:
        f, h = res.load_final_snapshot_and_halos()

    particles_inner = []
    for i in halo_IDs:
        r = get_a_fraction_virial_radius(i, fraction , f=f, h=h)

        # center on most bound ID and wrap so you avoid issues with halos at boundaries of box
        pynbody.analysis.halo.center(f[h[i].properties['mostboundID']], vel=False)
        f.wrap()

        # then you can center on halo's com
        pynbody.analysis.halo.center(h[i], vel=False)
        p = h[i][pynbody.filt.Sphere(r)]['iord']

        if percentage_min is not None:
            r_min = get_a_fraction_virial_radius(i, percentage_min)
            p_min = h[i][pynbody.filt.Sphere(r_min)]['iord']
            p = np.setdiff1d(p, p_min)

        particles_inner.extend(p)
    return particles_inner


############### LOAD RADII PROPERTIES ###############


def load_radii_properties(particles_class="in", feature_set="test",
                          in_ids_properties="/Users/lls/Documents/CODE/stored_files/all_out/radii_files"
                                            "/radii_properties_in_ids.npy",
                          out_ids_properties="/Users/lls/Documents/CODE/stored_files/all_out/radii_files/"
                                             "radii_properties_out_ids.npy"):
    if particles_class == "in":
        radius_properties = np.load(in_ids_properties)

    elif particles_class == "out":
        radius_properties = np.load(out_ids_properties)

    else:
        radius_properties = np.concatenate((np.load(in_ids_properties), np.load(out_ids_properties)))

    if feature_set == "training":
        ids, predicted_probabilities, true_labels = res.load_classification_results()
        index = ~np.in1d(radius_properties[:,0], ids)
        radius_particles = radius_properties[index,:]
        return radius_particles

    elif feature_set == "test":
        ids, predicted_probabilities, true_labels = res.load_classification_results()
        index = np.in1d(radius_properties[:,0], ids)
        radius_particles = radius_properties[index,:]
        return radius_particles

    else:
        return radius_properties


def extract_radii_properties(subset_ids, class_label, particles_type="all"):
    radii_in_properties = load_radii_properties(particles_class=class_label, feature_set=particles_type)
    index = np.in1d(radii_in_properties[:, 0], subset_ids)
    radii_properties_subset = radii_in_properties[index, :]
    return radii_properties_subset


def extract_radii_properties_subset_particles(particles="false negatives", EPS_predictions=None):
    ids, predicted_probabilities, true_labels = res.load_classification_results()

    if particles == "positives":
        radius_properties = load_radii_properties(particles_class="in", feature_set="test")
        return radius_properties

    elif particles == "negatives":
        radius_properties = load_radii_properties(particles_class="out", feature_set="test")
        return radius_properties

    else:
        if particles == "true positives":
            if EPS_predictions is True:
                particle_ids = EPS_pred.get_subset_classified_particles_EPS(particles=particles)
            else:
                particle_ids = res.get_true_positives(ids, predicted_probabilities, true_labels)
            particle_ids = particle_ids.astype('int')
            particles_true_class = "in"

        elif particles == "false negatives":
            if EPS_predictions is True:
                particle_ids = EPS_pred.get_subset_classified_particles_EPS(particles=particles)
            else:
                particle_ids = res.get_false_negatives(ids, predicted_probabilities, true_labels)
            particle_ids = particle_ids.astype('int')
            particles_true_class = "in"

        elif particles == "true negatives":
            if EPS_predictions is True:
                particle_ids = EPS_pred.get_subset_classified_particles_EPS(particles=particles)
            else:
                particle_ids = res.get_true_negatives(ids, predicted_probabilities, true_labels)
            particle_ids = particle_ids.astype('int')
            particles_true_class = "out"

        elif particles == "false positives":
            if EPS_predictions is True:
                particle_ids = EPS_pred.get_subset_classified_particles_EPS(particles=particles)
            else:
                particle_ids = res.get_false_positives(ids, predicted_probabilities, true_labels)
            particle_ids = particle_ids.astype('int')
            particles_true_class = "out"

        else:
            NameError("invalid set of particles.")

        radii_properties_subset = extract_radii_properties(particle_ids, particles_true_class)
        return radii_properties_subset


# def compare_virial_radius_subfind_pynbody(halo_id):
#     f, h = load_final_snapshot_and_halos()
#
#     virial_rad_subfind = h[halo_id].properties['rmean_200']*(10**3) / h[halo_id].properties['h'] * h[
#         halo_id].properties['a']
#     print("SUBFIND virial radius is " + str(virial_rad_subfind))
#
#     halo_centre_subfind = h[halo_id].properties['pos']*(10**3) / h[halo_id].properties['h'] * h[
#         halo_id].properties['a']
#     #f['pos'] -= halo_centre_subfind
#     virial_radius_pyn = pynbody.analysis.halo.virial_radius(h[halo_id], cen=halo_centre_subfind, overden=178)
#     print("pynbody virial radius is " + str(virial_radius_pyn))
#
#
# VIRIAL RADIUS FROM RMEAN_200 PROPERTY OF HALO
# def virial_radius(halo_id, f=None, h=None):
#     """ Return virial radius of the halo (in kpc)"""
#
#     if f is None and h is None:
#         f, h = load_final_snapshot_and_halos()
#
#     # rmean_200 is in units Mpc h**-1 a so need to convert it to kpc.
#     vir = h[halo_id].properties['rmean_200']*(10**3) / h[halo_id].properties['h'] * h[halo_id].properties['a']
#     vir = pynbody.array.SimArray(vir)
#     vir.units = 'kpc'
#     return vir


###################### FUNCTIONS TO HISTOGRAM SUBSET OF IDS AS A FUNCTION OF RADIUS IN THE HALO ######################


def get_num_particles_per_radius_bin(radius_particles, xscale="log", number_of_bins=20):

    if xscale == "log":
        bins = 10**np.linspace(np.log10(radius_particles.min()), np.log10(radius_particles.max()), number_of_bins)
        plt.xscale("log")
    elif xscale == "uniform":
        bins = number_of_bins
    elif xscale == "equal total particles":
        bins = get_log_spaced_bins_flat_distribution(radius_particles, number_of_bins_init=number_of_bins)
        plt.xscale("log")
    elif isinstance(xscale, (np.ndarray, list)):
        bins = xscale
    else:
        raise NameError("Choose either uniform or uniform-log scale.")

    n, bins = np.histogram(radius_particles, bins=bins)
    return n, bins


def plot_num_particles_per_radius_bin(radius_particles, yscale="log", line="-",
                                      xscale="log", label=None, color=None, number_of_bins=20, xlabel=None,
                                      ylabel=None, legend=None):

    num_particles, bins = get_num_particles_per_radius_bin(radius_particles, xscale=xscale,
                                                           number_of_bins=number_of_bins)
    mean_bins = np.array([np.mean([bins[i], bins[i + 1]]) for i in range(len(num_particles))])

    plt.plot(mean_bins, num_particles, label=label, color=color, linestyle=line)
    plt.scatter(mean_bins, num_particles, color=color)

    if xlabel is True:
        plt.xlabel(r"$r/r_{\mathrm{vir}}$")
    if ylabel is True:
        plt.ylabel(r"$N$")
    if legend is True:
        plt.legend(loc="best")

    if yscale == "log":
        plt.yscale("log")


def plot_scatter_line_vs_radius(bins, ratio, color=None, marker=None, label_scatter=None, label=None, linestyle=None,
                                xlabel=None,
                                ylabel=None, legend=None, legend_loc=None, title=None):
    plt.scatter(bins, ratio, color=color, marker=marker, label=label_scatter)
    l, = plt.plot(bins, ratio, color=color, label=label, linestyle=linestyle)

    if xlabel is True:
        plt.xlabel(r"$r/r_{\mathrm{vir}}$")

    if ylabel is True:
        plt.ylabel(r"$N/N_{\mathrm{all}}$")
    elif isinstance(ylabel, str):
        plt.ylabel(ylabel)

    if legend is True:
        if legend_loc is None:
            plt.legend(loc="best")
        else:
            plt.legend(loc=legend_loc)
    if title is not None:
        plt.title(title)
    return l,


def plot_ratio_ML_EPS_per_r_bins(rad_fraction_ML, rad_fraction_EPS, rad_fraction_all_in,
                                  xscale="equal total particles",
                                  number_of_bins=10, label=None, color=None, xlabel=None,
                                  ylabel=None, legend=None, legend_loc=None, title=None, marker='o',
                                  linestyle='-', label_scatter=None):

    n_all, bins_all = get_num_particles_per_radius_bin(rad_fraction_all_in, xscale=xscale,
                                                       number_of_bins=number_of_bins)
    n_ML, bins_ML = get_num_particles_per_radius_bin(rad_fraction_ML, xscale=bins_all,
                                                     number_of_bins=number_of_bins)
    n_EPS, bins_EPS = get_num_particles_per_radius_bin(rad_fraction_EPS, xscale=bins_all,
                                                       number_of_bins=number_of_bins)

    ratio = n_ML/ n_EPS
    mean_bins = np.array([np.mean([bins_all[i], bins_all[i+1]]) for i in range(len(ratio))])

    l, = plot_scatter_line_vs_radius(mean_bins, ratio, color=color, marker=marker, label_scatter=label_scatter,
                                     label=label, linestyle=linestyle,
                                     xlabel=xlabel, ylabel=ylabel, legend=legend, legend_loc=legend_loc, title=title)
    return l,


def plot_ratio_subset_vs_all_particles_per_r_bins(rad_fraction_all_in, rad_fraction_subset, xscale="log",
                                                  number_of_bins=10, label=None, color=None, xlabel=None,
                                                  ylabel=None, legend=None, legend_loc=None, title=None, marker='o',
                                                  linestyle='-'):

    n_all_log, bins_all = get_num_particles_per_radius_bin(rad_fraction_all_in, xscale=xscale,
                                                           number_of_bins=number_of_bins)
    n_FN_log, bins_all = get_num_particles_per_radius_bin(rad_fraction_subset, xscale=bins_all,
                                                          number_of_bins=number_of_bins)

    ratio = n_FN_log / n_all_log
    mean_bins = np.array([np.mean([bins_all[i], bins_all[i + 1]]) for i in range(len(ratio))])

    l, = plot_scatter_line_vs_radius(mean_bins, ratio, color=color, marker=marker, label=label, linestyle=linestyle,
                                     xlabel=xlabel, ylabel=ylabel, legend=legend,
                                     legend_loc=legend_loc, title=title)
    return l,


######################### EXTRACT RADIUS PROPERTIES FOR SUBSET PARTICLES #########################


def extract_radius_properties_particles_in_mass_bin(particles="true positives", mass_bin="high", EPS_predictions=None):
    bin_particles = mass.get_particles_in_mass_bin(mass_bin=mass_bin)
    ids_all, pred, true = res.load_classification_results()

    if particles == "true positives" or particles == "false negatives" or particles == "positives":
        label = 1
    elif particles == "true negatives" or particles == "false positives" or particles == "negatives":
        label = -1
    else:
        raise NameError("Enter a valid set of particles.")

    ids = ids_all[true == label]
    ids_in_bin = np.intersect1d(ids.astype('int'), bin_particles.astype('int'))

    particles_properties = extract_radii_properties_subset_particles(particles=particles,
                                                                     EPS_predictions=EPS_predictions)
    subset_index = np.in1d(particles_properties[:, 0], ids_in_bin)
    particles_bin_properties = particles_properties[subset_index, :]
    return particles_bin_properties


def extract_fraction_virial_radius_particles_in_mass_bin(particles="true positives", mass_bin="high",
                                                         EPS_predictions=None):
    particles_properties = extract_radius_properties_particles_in_mass_bin(particles=particles, mass_bin=mass_bin,
                                                                           EPS_predictions=EPS_predictions)
    fraction_virial_radius = np.array([particles_properties[:, 2][particles_properties[:, 2] != 0] if
                                    mass_bin == "small" else particles_properties[:, 2]]).flatten()
    return fraction_virial_radius