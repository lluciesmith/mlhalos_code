"""
:mod:`features`.

Feature extraction process for machine learning algorithm.
"""


import numpy as np
from sklearn.preprocessing import RobustScaler
import time

from . import density
from . import parameters
from . import window
from . import shear

########################################################################################################
# FEATURE EXTRACTION
#########################################################################################################


def extract_labeled_features(features_type="EPS trajectories",
                             initial_parameters=None, num_filtering_scales=50, rescale=None, n_samples=None,
                             density_contrast_in=None, density_contrast_out=None, add_EPS_label=False,
                             shear_scale=None, density_subtracted_shear=False, path=None, cores=10):
    """
    Extracts features of features_type given instance :class:`InitialConditionsParameters`.

    Args:
        features_type (str, list): Defines the type of features to extract from simulations data.
            "EPS trajectories" features are density values of particle's trajectory.
        initial_parameters (class): Instance of :class:'InitialConditionsParameters`. Default
            takes default args of class.
        density_contrast_in (ndarray): Optional input of 'in' particles' trajectories in form
            [n_particles, m_densities].
        density_contrast_out (ndarray): Optional input of 'out' particles' trajectories in form
            [n_particles, m_densities].
        num_filtering_scales (int): Number of top-hat smoothing filters to apply to the density field.
        add_EPS_label (bool): Add to the feature set the EPS prediction.
        rescale (str, None): None or "standard". Standard rescales features to zero mean and unit variance.
        n_samples (int, str): number of random samples of particles to form feature set.

    Returns:
        features (ndarray): Features of form [n_samples, m_features + label]. Ready to train
            the machine learning algorithm. NOTE: features are ordered such that features = np.concatenate(
            features_in, features_out) <--- IMPORTANT

    """
    if initial_parameters is None:
        parameters.InitialConditionsParameters(path=path)

    if type(features_type) == str:
        features = compute_features(features_type, initial_parameters, num_filtering_scales, rescale, n_samples,
                                      density_contrast_in, density_contrast_out, add_EPS_label, shear_scale,
                                      density_subtracted_shear, path=path, cores=cores)

    elif type(features_type) == list:
        # Need to allow the option for it to be a list of N feature types, not just two. (for now it's ok)

        features_1 = compute_features(features_type[0], initial_parameters, num_filtering_scales, rescale, n_samples,
                                      density_contrast_in, density_contrast_out, add_EPS_label, shear_scale,
                                      density_subtracted_shear, path=path, cores=cores)
        features_2 = compute_features(features_type[1], initial_parameters, num_filtering_scales, rescale, n_samples,
                                      density_contrast_in, density_contrast_out, add_EPS_label, shear_scale,
                                      density_subtracted_shear, path=path, cores=cores)

        assert all(features_1[:, -1] == features_2[:, -1])

        features = np.column_stack((features_1[:, :-1], features_2))
    else:
        raise TypeError("Features type can only be a string or a list of strings")

    return features


def compute_features(features_type, initial_parameters=None, num_filtering_scales=50, rescale=None, n_samples=None,
                     density_contrast_in=None, density_contrast_out=None, add_EPS_label=False,
                     shear_scale=None, density_subtracted_shear=False, path=None, cores=10):
    if initial_parameters is None:
        parameters.InitialConditionsParameters(path=path)

    if features_type == "EPS trajectories":
        features = get_density_features(initial_parameters, num_filtering_scales, density_contrast_in,
                                        density_contrast_out, n_samples, add_EPS_label, path=path)

    elif features_type == "shear":
        t0 = time.clock()
        t00 = time.time()

        features = get_shear_features(initial_parameters, num_filtering_scales, shear_scale,
                                      density_subtracted_shear, path=path, cores=cores)

        print("Wall time" + str(time.time() - t00))
        print("Process time" + str(time.clock() - t0))

    else:
        raise TypeError("Not a valid feature type.")

    if rescale == "standard":
        features = rescale_features(features)

    return features


def rescale_features(features):
    """
    Rescale features to have zero mean and unit variance.

    This function uses the inbuilt :func:`RobustScaler` from :mod:`sklearn.preprocessing`
    of the scikit-learn package. This function is preferred to :func:`StandardScaler` since
    it is not sensitive to outliers in feature distributions.
    """
    rescaled_features = features[:, :-1]
    rescaled_features = RobustScaler().fit_transform(rescaled_features)
    rescaled_features_with_label = np.column_stack((rescaled_features, features[:, -1]))
    return rescaled_features_with_label


########################################################################################################

# FEATURE EXTRACTION FOR FEATURES_TYPE = "SHEAR"

# Features of type "shear" are given by the two eigenvalues of the shear tensor

#########################################################################################################

def get_shear_features(initial_parameters, num_filtering_scales, shear_scale, density_subtracted_shear, path=None,
                       cores=10):

    sp = shear.ShearProperties(initial_parameters=initial_parameters, num_filtering_scales=num_filtering_scales,
                               shear_scale=shear_scale, path=path, number_of_processors=cores)

    if density_subtracted_shear is False:
        features = get_features_from_shear_ellipticity_and_prolateness(sp.ellipticity, sp.prolateness,
                                                                       initial_parameters=initial_parameters)
    else:
        features = get_features_from_shear_ellipticity_and_prolateness(sp.density_subtracted_ellipticity,
                                                                       sp.density_subtracted_prolateness,
                                                                       initial_parameters=initial_parameters)
    return features


def get_ids_type_shear_property(initial_parameters, shear_property, ids_type='in'):

    if ids_type == 'in':
        ids_type = initial_parameters.ids_IN
    elif ids_type == 'out':
        ids_type = initial_parameters.ids_OUT
    elif isinstance(ids_type, (np.ndarray, list)):
        ids_type = ids_type
    else:
        raise TypeError("Invalid subset of ids")

    shear_property_of_ids = np.array([shear_property[particle_id] for particle_id in ids_type])
    return shear_property_of_ids


def get_in_out_ellipticity(initial_parameters, ellipticity):
    ellipticity_in = get_ids_type_shear_property(initial_parameters, ellipticity, ids_type='in')
    ellipticity_out = get_ids_type_shear_property(initial_parameters, ellipticity, ids_type='out')
    return ellipticity_in, ellipticity_out


def get_in_out_prolateness(initial_parameters, prolateness):
    prolateness_in = get_ids_type_shear_property(initial_parameters, prolateness, ids_type='in')
    prolateness_out = get_ids_type_shear_property(initial_parameters, prolateness, ids_type='out')
    return prolateness_in, prolateness_out


def get_features_from_shear_ellipticity_and_prolateness(ellipticity, prolateness,
                                                        initial_parameters=None):
    if initial_parameters is None:
        initial_parameters = parameters.InitialConditionsParameters()

    ellipticity_in, ellipticity_out = get_in_out_ellipticity(initial_parameters, ellipticity)
    prolateness_in, prolateness_out = get_in_out_prolateness(initial_parameters, prolateness)

    assert len(ellipticity_in) == len(prolateness_in)
    assert len(ellipticity_out) == len(prolateness_out)

    in_label = np.ones(len(ellipticity_in))
    out_label = (np.ones(len(ellipticity_out))) * -1

    features_in = np.column_stack((ellipticity_in, prolateness_in, in_label))
    features_out = np.column_stack((ellipticity_out, prolateness_out, out_label))

    shear_features = np.concatenate((features_in, features_out))
    return shear_features


########################################################################################################

# FEATURE EXTRACTION FOR FEATURES_TYPE = "EPS TRAJECTORIES"

# Features of type "EPS trajectories" are given by the particles' trajectories, i.e. the density contrasts obtained
# from the particles' density, smoothed by top-hat filters of increasing radius.

#########################################################################################################


def get_density_features(initial_parameters, num_filtering_scales, density_contrast_in=None,
                         density_contrast_out=None, n_samples=None, add_EPS_label=False, path=None):

    if density_contrast_in is None and density_contrast_out is None:
        density_contrasts = density.DensityContrasts(initial_parameters=initial_parameters,
                                                     num_filtering_scales=num_filtering_scales, path=path)
        delta_in = density_contrasts.density_contrast_in
        delta_out = density_contrasts.density_contrast_out

    elif density_contrast_in is not None and density_contrast_out is not None:
        delta_in = density_contrast_in
        delta_out = density_contrast_out

    else:
        raise TypeError("Please be consistent in specifying density contrasts for in and out particles.")

    # Optional feature: EPS predicted label based on "trajectory crossing"

    if add_EPS_label is True:
        delta_in, delta_out = add_EPS_label_as_a_feature(initial_parameters, num_filtering_scales,
                                                         delta_in, delta_out)

    # From trajectories extract features.

    if n_samples is None:
        features = get_features_from_delta_in_out(delta_in, delta_out)

    else:
        features = get_features_from_subset_delta_in_out(delta_in, delta_out, n_samples=n_samples)

    return features


def label_density_contrasts_in_or_out(delta_in, delta_out):
    """Label in particles +1 and out particles -1."""

    in_label = np.ones(len(delta_in))
    out_label = (np.ones(len(delta_out))) * -1

    delta_in_with_label = np.column_stack((delta_in, in_label))
    delta_out_with_label = np.column_stack((delta_out, out_label))

    return delta_in_with_label, delta_out_with_label


def all_density_contrasts_with_label(delta_in_with_label, delta_out_with_label):
    """
    Set a unique ndarray of density contrasts for both in and out particles
    with appropriate labels of form [n_particles, m_density_contrasts + label].
    """
    delta_all_with_label = np.concatenate((delta_in_with_label, delta_out_with_label))
    return delta_all_with_label


def subset_of_n_random_density_contrasts(delta_all, n=2000):
    n_random_delta = delta_all[np.random.choice(range(len(delta_all)), n, replace=False)]
    return n_random_delta


def get_features_from_delta_in_out(delta_in, delta_out):
    """
    Translate density contrasts of in and out particles into features.

    Args:
        delta_in (ndarray): Density contrasts of in particles.
        delta_out (ndarray): Density contrasts of out particles.
    Returns:
        features (ndarray): Features of form [n_samples, m_features + label]. Ready to train
            the machine learning algorithm.

    """
    delta_in_with_label, delta_out_with_label = label_density_contrasts_in_or_out(delta_in, delta_out)
    features = all_density_contrasts_with_label(delta_in_with_label, delta_out_with_label)

    return features


def get_features_from_subset_delta_in_out(delta_in, delta_out, n_samples=2000):
    """
    Extract features for n_samples randomly picked particles given density contrasts of all in and out particles.
    """
    delta_all_with_label = get_features_from_delta_in_out(delta_in, delta_out)
    features = subset_of_n_random_density_contrasts(delta_all_with_label, n=n_samples)

    return features


def number_in_and_out_ids_in_random_subset(n_random_delta, delta_in, delta_out):
    """Returns number of in and out particles in feature set's random sample."""

    randomset = set([tuple(x) for x in n_random_delta])
    in_set = set([tuple(x) for x in delta_in])
    out_set = set([tuple(x) for x in delta_out])

    number_of_in_chosen = len(np.array([x for x in in_set & randomset]))
    number_of_out_chosen = len(np.array([x for x in out_set & randomset]))

    return number_of_in_chosen, number_of_out_chosen

def add_EPS_label_as_a_feature(initial_parameters, num_filtering_scales, delta_in, delta_out):
    """
    The EPS prediction is +1 for particles whose trajectory crosses the density threshold at mass scales within the
    mass range of "in" halos, otherwise -1.

    """

    w = window.WindowParameters(initial_parameters=initial_parameters, num_filtering_scales=num_filtering_scales)
    mass_scales = w.smoothing_masses
    min_mass = initial_parameters.halo[initial_parameters.max_halo_number]['mass'].sum()
    max_mass = initial_parameters.halo[initial_parameters.min_halo_number]['mass'].sum()
    in_scales = np.intersect1d(np.where(mass_scales >= min_mass), np.where(mass_scales <= max_mass))

    EPS_label_in = np.array([1 if any(num >= 1.0169 for num in delta_in[i, in_scales]) else -1 for i in
                             range(len(delta_in))])
    delta_in = np.column_stack((delta_in, EPS_label_in))

    EPS_label_out = np.array([1 if any(num >= 1.0169 for num in delta_out[i, in_scales]) else -1 for i in
                              range(len(delta_out))])
    delta_out = np.column_stack((delta_out, EPS_label_out))

    return delta_in, delta_out
