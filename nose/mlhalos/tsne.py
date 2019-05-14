"""
:mod:`tsne`

Returns t-SNE plot given a set of features.
"""

import numpy as np
from sklearn.manifold import TSNE

from . import plot


def reduce_features_in_two_dimensions_using_tsne(density_contrasts_with_label,
                                                 n_components=2, perplexity=50.0,
                                                 early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000,
                                                 metric="euclidean", init="random",
                                                 random_state=None):
    """
    Tune t-SNE hyperparameters and reduce feature space to two dimensions using t-SNE.
    We use the t-SNE algorithm inbuilt in the scikit-learn package.

    Args:
        density_contrasts_with_label (ndarray): labeled density contrasts for in and out particles.
        n_components (int): Dimensions of reduced feature space. Default is 2.
        perplexity (float): Related to number of nearest neighbours used in other "manifold" algorithms.
           t-SNE not very sensitive to this. Usually in the range 5-50. Default is 50.
        early_exaggeration (float): Controls how tight clusters in original space will be in embedded space.
           t-SNE not very sensitive to this. Default is 4.0.
        learning_rate (float): This can be critical. Should be between 100 and 1000. Default is 1000.
        n_iter (int): Max number of iteration for optimization. Default is 1000.
        metric (str): Metric of similarity in feature space. Default is "euclidean", which is
             squared euclidean distance.
        init (str):"random" or "pca". Default is "random".
        random_state: Psuedo Random Number generator seed control.

    Returns:
        labeled_features_two_dimensions (2Darray): Two-dimensional features array which features as columns.
    """
    if len(density_contrasts_with_label[0]) > 52:
        raise ValueError("t-SNE dimensionality reduction doesn't work well with more than 50 features. Perform "
                         "further dimensionality reduction (e.g.PCA) before t-SNE plot.")

    tsne_algo = TSNE(n_components=n_components, perplexity=perplexity,
                                early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=n_iter,
                                metric=metric, init=init,
                                random_state=random_state)

    y = tsne_algo.fit_transform(density_contrasts_with_label[:, :-1])
    labeled_features_two_dimensions = np.column_stack((y, density_contrasts_with_label[:, -1]))
    return labeled_features_two_dimensions


def split_features_in_and_out(features_with_label):
    """Split labeled features ndarray in two ndarrays, features of 'in' particles and features of 'out' particles."""

    features_in = features_with_label[features_with_label[:, -1] == 1]
    features_out = features_with_label[features_with_label[:, -1] == -1]
    return features_in, features_out


def get_tsne_plot(features_with_label, n_components=2, perplexity=50.0,
                  early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000,
                  metric="euclidean", init="random",
                  random_state=None, title=False):
    """
    Returns t-sne plot given a set of labeled features and t-SNE hyperparameters.
    It reduces the feature space to two-dim (see :func:`reduce_features_in_two_dimensions_using_tsne`),
    it splits the features in the two classes (see :func:`split_features_in_and_out`)
    and it gives the t-SNE plot using :func:`plot_tsne_features` in :mod:`plot`.
    """
    features_tsne = reduce_features_in_two_dimensions_using_tsne(features_with_label,
                                                                 n_components=n_components, perplexity=perplexity,
                                                                 early_exaggeration=early_exaggeration,
                                                                 learning_rate=learning_rate, n_iter=n_iter,
                                                                 metric=metric, init=init,
                                                                 random_state=random_state)
    features_in, features_out = split_features_in_and_out(features_tsne)

    fig = plot.plot_tsne_features(features_in, features_out)
    number_of_particles = len(features_tsne)
    number_of_filters = features_with_label.shape[1]-1

    if title is True:
        fig.suptitle("t-SNE of " + str(number_of_filters) +
                     " features for " + str(number_of_particles) + " particles")
    return fig
