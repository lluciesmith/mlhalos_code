import sklearn.ensemble.forest as forest
from sklearn.ensemble.forest import _generate_sample_indices
import numpy as np


def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None):

    """Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples)
        tree.used_indices = indices
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight is not None:
            raise RuntimeError("not compatible with the hacked parallel_build_trees")

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree


def install_scikit_hack():
    forest._parallel_build_trees = _parallel_build_trees
