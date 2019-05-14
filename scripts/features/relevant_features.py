"""
Take relevant features - set feature number as sys.argv[1]

take subset of particles - out particles, small-mass bin --> inner/outer
                            out particles, high-mass bin ---> inner/outer

"""
# import matplotlib
# matplotlib.use('macosx')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code')
from utils import classification_results as res
from utils import radius_func as rf
from mlhalos import distinct_colours


def plot_feature_distributions_positives_inner_outer(relevant_feature_index, mass_bin, threshold_inner_outer, normed,
                                                     all_features=None):
    if all_features is None:
        features_test = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/features_test.npy")
    else:
        features_test = all_features

    relevant_feature = features_test[:, int(relevant_feature_index)]

    # get particles POSITIVES VS NEGATIVES

    if mass_bin == "":
        positives_r_prop = rf.extract_radii_properties_subset_particles(particles="positives")
    else:
        positives_r_prop = rf.extract_radius_properties_particles_in_mass_bin(particles="positives", mass_bin=mass_bin)

    positives_r_prop = positives_r_prop[np.logical_and(positives_r_prop[:, 2] != 0, positives_r_prop[:, 1] > 25.6)]

    pos_inner = positives_r_prop[:, 0][positives_r_prop[:, 2] < float(threshold_inner_outer)]
    pos_outer = positives_r_prop[:, 0][positives_r_prop[:, 2] >= float(threshold_inner_outer)]

    # get features

    out_f = relevant_feature[features_test[:, -1] == -1]

    ids, pred, true = res.load_classification_results()
    f_inner = relevant_feature[np.in1d(ids, pos_inner)]
    f_outer = relevant_feature[np.in1d(ids, pos_outer)]

    # plot

    colors = distinct_colours.get_distinct(4)
    fig = plt.figure()
    plt.hist(out_f, bins=10, color=colors[0], label="out", histtype='step', normed=normed, linewidth=1.5)
    plt.hist(f_inner, bins=10, color=colors[1], label="in-inner", histtype='step', normed=normed, linewidth=1.5)
    plt.hist(f_outer, bins=10, color=colors[3], label="in-outer", histtype='step', normed=normed, linewidth=1.5)
    plt.legend(loc=2)
    plt.xlabel("Feature " + str(relevant_feature_index))
    plt.ylabel(r"$N$")
    if mass_bin != "":
        plt.title(str(mass_bin).title() + "-mass bin positives")
    return fig


if __name__ == "__main__":

    relevant_feature_index = np.arange(50)
    mass_bins = ["high", "mid", "small"]
    threshold_inner_outer = 0.4
    normed = True
    # save = sys.argv[5]

    features_test = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/features_test.npy")
    for mass_bin in mass_bins:
        for index in relevant_feature_index:
            figure = plot_feature_distributions_positives_inner_outer(index, mass_bin, threshold_inner_outer, normed,
                                                                      all_features=features_test)
            if index < 27:
                plt.xlim(0.7, 1.2)
            else:
                plt.xlim(0.9, 1.1)

            figure.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feature_distributions/" + str(mass_bin) +
                           "_mass/feature_" + str(index) + "_" + str(mass_bin)+ "_mass_bin.pdf", bbox_inches='tight')
            figure.clf()

    # if isinstance(relevant_feature_index, int):
    #     figure = plot_feature_distributions_positives_inner_outer(relevant_feature_index, mass_bin, threshold_inner_outer,
    #                                                               normed, all_features=features_test)
    #     if save is True:
    #         figure.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feature_distributions/" + str(
    #                        mass_bin) + "_mass/feature_" + str(relevant_feature_index) + "_" + str(mass_bin) +
    #                        "_mass_bin.pdf", bbox_inches='tight')
    #     else:
    #         figure.show()
    #
    # elif isinstance(relevant_feature_index, (list, np.ndarray)):
    #     for index in relevant_feature_index:
    #         figure = plot_feature_distributions_positives_inner_outer(index, mass_bin, threshold_inner_outer, normed)
    #
    #         if save is True:
    #             figure.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feature_distributions/" + str(mass_bin)
    #                            + "_mass/feature_" + str(index) + "_" + str(mass_bin)+ "_mass_bin.pdf",
    #                            bbox_inches='tight')
    #         else:
    #             figure.show()
    #
    #         figure.clf()
    # else:
    #     NameError("Not a valid index for the features")


# features = np.load("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/features_all_particles.npy")
# index_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy")
# index_test = np.setdiff1d(np.arange(len(features)), index_training)
# features_test = features[index_test.astype('int')]
# np.save("/Users/lls/Documents/CODE/stored_files/all_out/not_rescaled/features_test.npy", features_test)

# features = np.load("/Users/lls/Documents/CODE/stored_files/all_out/features_rescaled_full_mass.npy")
# index_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy")
# index_test = np.setdiff1d(np.arange(len(features)), index_training)
# features_test = features[index_test.astype('int')]
# np.save("/Users/lls/Documents/CODE/stored_files/all_out/features_test.npy", features_test)