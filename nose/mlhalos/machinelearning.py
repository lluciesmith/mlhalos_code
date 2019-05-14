"""
:mod:`machinelearning`

Trains machine learning algorithm given a set of labeled features and computes its ROC curve and
AUC score from prediction scores.
"""
from __future__ import division
import numpy as np
import subprocess
from scipy.integrate import trapz
import time

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import export_graphviz
from sklearn.externals import joblib

from . import plot
from . import window
from . import forest_fix


class MLAlgorithm(object):
    """ Trains machine learning algorithm given set of labeled features."""

    def __init__(self, labeled_features, algorithm='Random Forest', split_data_method='train_test_split',
                 train_size=0.8, cross_validation=True, num_cv_folds=5, tree_ids=False, n_jobs=1, save=False,
                 path="classifier.pkl", method="classification", param_grid=None):
        """
        Instantiates :class:`MLAlgorithm` given set of labeled features and choice of algorithm.

        Args:
            labeled_features (ndarray): Set of labeled features of form [n_samples, n_features + label].
                [n_samples, n_features + label].
            algorithm (str): Choice of machine learning algorithm. Default is Random Forest.
            split_data_method (str, None): "train_test_split" (default) or "KFold". Optional choice of
                method to split features into training and testing sets.
            method (str): Either classification or regression

        Returns:
            algorithm: Trained machine learning algorithm.
            y_predicted (array): Predicted labels of testing set.
            y_proba_predicted (array): Predicted class probabilities of test set in form [n_samples, n_classes].
                The order of the classes corresponds to that in the attribute classes_ of the algorithm.
            y_true (array): True labels of testing set.
            name_algorithm: Name of machine learning algorithm used.

        """

        self.name_algorithm = algorithm
        self.method = method
        self.cross_validation = cross_validation
        self.num_cv_folds = num_cv_folds
        self.tree_ids = tree_ids
        self.n_jobs = n_jobs
        self.param_grid = param_grid

        if split_data_method == "train_test_split":
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(labeled_features,
                                                                                         split_data_method,
                                                                                         train_size=train_size)

        else:
            self.X_train = labeled_features[:, :-1]
            self.y_train = labeled_features[:, -1]

            self.X_test = None
            self.y_test = None

        ts = time.time()
        self.fit_algorithm()
        te = time.time()
        print("Time to train the machine learning algorithm (seconds): " + str(te - ts))
        self.algo = self.algorithm

        if self.X_test is not None:
            self.predict_test_set()

        if method == "classification":
            self.classifier = self.algorithm

        if save is True:
            joblib.dump(self.algorithm, path)

    def fit_algorithm(self):
        self.algorithm = self.train_algorithm(algorithm=self.name_algorithm, method=self.method,
                                              cross_validation=self.cross_validation, cv=self.num_cv_folds,
                                              tree_ids=self.tree_ids, n_jobs=self.n_jobs, param_grid=self.param_grid)

        if self.cross_validation is True:
            self.trees = self.algorithm.best_estimator_.estimators_
            self.feature_importances = self.algorithm.best_estimator_.feature_importances_
            self.best_estimator = self.algorithm.best_estimator_
        else:
            self.trees = self.algorithm.estimators_
            self.feature_importances = self.algorithm.feature_importances_

        self.indices_importances = np.argsort(self.feature_importances)[::-1]

    def predict_test_set(self):
        if self.method == "classification":
            self.predicted_proba_test = self.algorithm.predict_proba(self.X_test)
        elif self.method == "regression":
            self.predicted_proba_test = self.algorithm.predict(self.X_test)

        self.true_label_test = self.y_test

    @staticmethod
    def split_train_test(features_with_output_label, split_data_method, train_size=0.8):
        """
        Splits features in training and testing sets given split_data_method.
        Options are "train_test_split" (default) or "KFold".
        """
        X = features_with_output_label[:, :-1]
        y = features_with_output_label[:, -1]

        if split_data_method == 'train_test_split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        elif split_data_method == 'KFold':
            kf_iterator = KFold(len(X), n_folds=10, shuffle=False)
            for train_index, test_index in kf_iterator:
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        else:
            raise NameError("Invalid split data method selected.")

        return X_train, X_test, y_train, y_test

    def train_algorithm(self, algorithm="Random Forest", method="classification", cross_validation=True, cv=5,
                        tree_ids=False, n_jobs=1, param_grid=None):
        trained_algo = self.tune_hyperparameters(algorithm=algorithm, method=method, cross_validation=cross_validation,
                                              cv=cv, tree_ids=tree_ids, n_jobs=n_jobs, param_grid=param_grid)
        trained_algo = trained_algo.fit(self.X_train, self.y_train)
        return trained_algo

    def tune_hyperparameters(self, algorithm="Random Forest",method="classification", cross_validation=True,
                             cv=5, tree_ids=False, n_jobs=1, param_grid=None):
        """
        Tunes hyperparameters of algorithm using a scikit-learn inbuilt cross validation method,
        :func:`sklearn.model_selection.GridSearchCV`.

        """

        if algorithm == "Random Forest":
            if method == "classification":

                if tree_ids is True:
                    forest_fix.install_scikit_hack()

                if cross_validation is True:
                    classifier = self.cross_validate_rf_classifier(n_jobs, cv, param_grid=param_grid)

                elif cross_validation is False:
                    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                        min_impurity_split=1e-07, min_samples_leaf=20,
                                                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                        n_estimators=400, n_jobs=60, oob_score=False, random_state=None,
                                                        verbose=0, warm_start=False)
                else:
                    raise NameError("Select true or false for cross-validation argument")

                return classifier

            elif method == "regression":
                if cross_validation is True:
                    # work out cross validation for random forest regressor
                    regr = self.cross_validate_rf_regression(n_jobs, cv, param_grid=param_grid)

                elif cross_validation is False:
                    # hardcode some hyperparameters you think are sensible
                    regr = RandomForestRegressor(n_estimators=400, max_features=15, min_samples_leaf=15,
                                                 criterion="mse")

                else:
                    raise NameError("Select true or false for cross-validation argument")
                return regr

            else:
                raise NameError("Choose whether to use classification or regression")
        else:
            raise NameError("Invalid machine learning algorithm")

    # Random Forest Classifier functions

    def cross_validate_rf_classifier(self, n_jobs, cv, param_grid=None):
        if param_grid is None:
            param_grid = {"n_estimators": [400, 600],
                          "max_features": ["auto", 0.4],
                          "min_samples_leaf": [15, 20],
                          "criterion": ["gini", "entropy"]
                          }

        random_forest = RandomForestClassifier(bootstrap=True, min_samples_split=2, n_jobs=n_jobs)
        auc_scorer = self.AucScore()
        classifier = GridSearchCV(random_forest, param_grid, scoring=auc_scorer, cv=cv, n_jobs=1)
        # random_forest = RandomForestClassifier(n_estimators=1000, min_samples_leaf=15,
        #                                        criterion="entropy", bootstrap=True,
        #                                        min_samples_split=2, n_jobs=n_jobs)
        # param_grid = {"max_features": ["auto", "log2", 0.3, 0.2, 0.15, 0.4, 0.5, 0.6],
        #               }
        # auc_scorer = self.AucScore()
        # classifier = grid_search.GridSearchCV(random_forest, param_grid, scoring=auc_scorer, cv=cv,
        #                                       n_jobs=1)
        return classifier

    class AucScore(object):
        def __call__(self, estimator, X, y, true_class=1):
            y_probabilities = estimator.predict_proba(X)
            return get_auc_score(y_probabilities, y, true_class=true_class)

    # Random Forest Regressor functions

    @staticmethod
    def cross_validate_rf_regression(n_jobs, cv, param_grid=None):
        if param_grid is None:
            param_grid = {"n_estimators": [800, 1000, 1300],
                          "max_features": [20, "sqrt", 50],
                          "min_samples_leaf": [5, 15],
                          #"criterion": ["mse", "mae"],
                          }

        random_forest = RandomForestRegressor(bootstrap=True, n_jobs=n_jobs)
        regr_rf = GridSearchCV(random_forest, param_grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=1)
        return regr_rf


def get_top_level_split_feature(tree_classifier, feature_ID, max_nodes_level=10000):
    features_per_level = get_features_per_tree_level(tree_classifier, max_nodes_level=max_nodes_level)

    if features_per_level[0] == feature_ID:
        top_level = 0

    else:
        for level in range(1, len(features_per_level)):
            if any(num == feature_ID for num in features_per_level[level]):
                break
            else:
                NameError("Increase the max_nodes_level of the tree since feature " + str(feature_ID) + " is not found "
                          "at tree level with nodes < " + str(max_nodes_level))
        top_level = level

    return top_level


def get_features_per_tree_level(tree_classifier, max_nodes_level=10000):
    """
    Args:
        tree_classifier: DecisionTreeClassifier estimator.
        max_nodes_level: Maximum number of nodes per level to compute this analysis. The higher this number,
            the more in depth you will search in the tree, but the more computationally expensive it will be.

    Returns:
        array of shape [n_levels, n_features], where row i includes features used at level i of the tree.
        Level 0 includes node 0 only, the root of the tree.

    """
    tree = tree_classifier.tree_

    features_per_level = []
    features_per_level.append(tree.feature[0])

    nodes = np.array([0])
    children_number = len(children_nodes(tree, nodes))

    while children_number < max_nodes_level:
        nodes = children_nodes(tree, nodes)
        features = children_features(tree, nodes)
        features_per_level.append(features)

        children_number = len(children_nodes(tree, nodes))
        # print(children_number)

    return features_per_level


def children_nodes(tree, nodes):
    assert isinstance(nodes, np.ndarray), "Nodes argument needs to be an array"

    node_right = [tree.children_right[node] for node in nodes]
    node_left = [tree.children_left[node] for node in nodes]
    return np.array((node_right, node_left)).flatten()


def children_features(tree, nodes):
    features = [tree.feature[node] for node in nodes]
    return features


def get_particle_ids_test_set(features, X_test, initial_parameters):
    """

    Args:
        features (array): set of labeled features that trained the algorithm.
        X_test (array): Subsample of features kept out of the training for the testing.
        initial_parameters (class): initial parameters of the problem.

    Returns:
        test_particles (list): List of particle ids in the test set.
    """

    ids = np.concatenate((initial_parameters.ids_IN, initial_parameters.ids_OUT))
    index = get_row_index_features_matrix_equiv_test_set(features, X_test)

    test_particles = ids[index]
    return test_particles


def get_row_index_features_matrix_equiv_test_set(features, X_test):
    index = []
    for j in range(len(X_test)):
        ind = np.where([np.allclose(X_test[j], features[i, :-1]) for i in range(len(features))])
        index.append(ind[0][0])

    return index


def classify_data(algorithm, X_new, proba=True):
    """
    Predicts label of new set of samples using trained algorithm.

    Args:
        algorithm: Trained classifier machine learning algorithm.
        X_new: New set of samples.

    Returns:
        labels (list): Class labels of new set of particles.
            +1 labels in particles and -1 labels out particles.

    """
    ts = time.time()
    if proba is True:
        labels = algorithm.predict_proba(X_new)
    else:
        labels = algorithm.predict(X_new)
    te = time.time()
    print("Time to make predictions (seconds): " + str(te - ts))
    return labels


def get_auc_score(y_proba, y_true, true_class=1):
    fpr, tpr, auc, threshold = roc(y_proba, y_true, true_class=true_class)
    return auc


def roc(y_proba, y_true, true_class=1):
    """
    Produce the false positive rate and true positive rate required to plot a ROC curve. Compute the Area Under the
    Curve(auc) using the trapezoidal rule.
    True class is 'in' label.

    Args:
        y_proba (np.array): An array of probability scores, either a 1d array of size N_samples or an nd array,
            in which case the column corresponding to the true class will be used.
            You can produce array of probability scores by doing predict_with_proba and then for each sample you get
            its probability to be in and its probability to be out. (Should be column_0 = in and column_1 = out). You
            can give this array checking that true_class is 'in' class probabilities.).
        y_true (np.array): An array of class labels, of size (N_samples,)
        true_class (int): Which class is taken to be the "true class". Default is 1.

    Returns:
        fpr (array): An array containing the false positive rate at each probability threshold
        tpr (array): An array containing the true positive rate at each probability threshold
        auc (array): The area under the ROC curve.

    Raises:
        IOError: "Predicted and true class arrays are not same length."

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    """
    if len(y_proba) != len(y_true):
        raise IOError("Predicted and true class arrays are not same length.")

    if len(y_proba.shape) > 1:

        if true_class == 1:
            proba_in_class = y_proba[:, 1]  # The order of classes in y_proba is first column - 1 and second +1
        elif true_class == -1:
            proba_in_class = y_proba[:, 0]

    else:
        proba_in_class = y_proba

    # 50 evenly spaced numbers between 0,1.
    threshold = np.linspace(0., 1., 50)

    # This creates an array where each column is the prediction for each threshold. It checks if predicted probability
    # of being "in" is greater than probability threshold. If yes it returns True, if no it returns False.
    preds = np.tile(proba_in_class, (len(threshold), 1)).T >= np.tile(threshold, (len(proba_in_class), 1))

    # Make y_true a boolean vector - array that returns True if particle is "in" and False if particle is "out".
    # It is true for all values of threshold ( hence it is rearranged as (len(threshold),1).T).
    y_bool = (y_true == true_class)
    y_bool = np.tile(y_bool, (len(threshold), 1)).T

    # These arrays compare predictions to Y_bool array at each threshold. Sum(axis=0) counts the number of "True"
    # samples at each threshold.
    # If both the predictions and Y_bool return true (or false) it is a true positive (or true negative).
    # If predictions return true and y_bool is false, then it is a false positive.
    # If predictions return false and y_book is true, then it is a false negative.
    TP = (preds & y_bool).sum(axis=0)
    FP = (preds & ~y_bool).sum(axis=0)
    TN = (~preds & ~y_bool).sum(axis=0)
    FN = (~preds & y_bool).sum(axis=0)

    # True positive rate is defined as true positives / (true positives + false negatives), i.e. true positives out
    # of all positives. False positive rate is defined as false positives / (false positives + true negatives),
    # i.e. false positives out of all negatives.
    tpr = np.zeros(len(TP))
    tpr[TP != 0] = TP[TP != 0] / (TP[TP != 0] + FN[TP != 0])
    fpr = FP / (FP + TN)  # Make sure you have included from __future__ import division for this if using python 2!

    # Reverse order of fpr and tpr so that thresholds go from high to low.
    fpr = np.array(fpr)[::-1]
    tpr = np.array(tpr)[::-1]
    threshold = threshold[::-1]

    # Compute Area Under Curve according to trapezoidal rule, using :func:`trapz` in :mod:`scipy.integrate`.
    auc = trapz(tpr, fpr)

    return fpr, tpr, auc, threshold


def get_roc_curve(y_proba, y_true, true_class=1, label=[" "]):
    """Get ROC plot given predicted probability scores of classes and true classes for samples."""
    fpr, tpr, auc, threshold = roc(y_proba, y_true, true_class=true_class)
    plot.roc_plot(fpr, tpr, auc, labels=label)


def false_negative_rate(y_proba, y_true, true_class=1):
    """
    Produce the false negative rate.
    True class is 'in' label.

    Args:
        y_proba (np.array): An array of probability scores, either a 1d array of size N_samples or an nd array,
            in which case the column corresponding to the true class will be used.
            You can produce array of probability scores by doing predict_with_proba and then for each sample you get
            its probability to be in and its probability to be out. (Should be column_0 = in and column_1 = out). You
            can give this array checking that true_class is 'in' class probabilities.).
        y_true (np.array): An array of class labels, of size (N_samples,)
        true_class (int): Which class is taken to be the "true class". Default is 1.

    Returns:
        fnr (array): An array containing the false negative rate at each probability threshold

    Raises:
        IOError: "Predicted and true class arrays are not same length."

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to ``fnr``,
    which is sorted in reversed order during its calculation.

    """
    if len(y_proba) != len(y_true):
        raise IOError("Predicted and true class arrays are not same length.")

    if len(y_proba.shape) > 1:

        if true_class == 1:
            proba_in_class = y_proba[:, 1]  # The order of classes in y_proba is first column - 1 and second +1
        elif true_class == -1:
            proba_in_class = y_proba[:, 0]

    else:
        proba_in_class = y_proba

    # 50 evenly spaced numbers between 0,1.
    threshold = np.linspace(0., 1., 50)

    # This creates an array where each column is the prediction for each threshold. It checks if predicted probability
    # of being "in" is greater than probability threshold. If yes it returns True, if no it returns False.
    preds = np.tile(proba_in_class, (len(threshold), 1)).T > np.tile(threshold, (len(proba_in_class), 1))

    # Make y_true a boolean vector - array that returns True if particle is "in" and False if particle is "out".
    # It is true for all values of threshold ( hence it is rearranged as (len(threshold),1).T).
    y_bool = (y_true == true_class)
    y_bool = np.tile(y_bool, (len(threshold), 1)).T

    TP = (preds & y_bool).sum(axis=0)
    FN = (~preds & y_bool).sum(axis=0)
    fnr = FN / (FN + TP)  # Make sure you have included from __future__ import division for this if using python 2!

    # Reverse order of fpr and tpr so that thresholds go from high to low.
    fnr = np.array(fnr)[::-1]
    threshold = threshold[::-1]

    return threshold, fnr


def overfitting_check(ml_algorithm):
    """
    Returns an assertion error if auc score of validation set (during cross-validation) and auc score of test set
    differ more than 0.05.

    Args:
        ml_algorithm: Instance of :class:`MLAlgorithm`

    Raises:
        AssertionError: "Risk of overfitting since auc scores of validation set and test set differ significantly."

    """
    auc_val_score = ml_algorithm.classifier.best_score_
    fpr, tpr, auc_test_score, threshold = roc(ml_algorithm.y_proba_predicted, ml_algorithm.y_true)

    assert np.isclose(auc_val_score, auc_test_score, atol=0.05), \
        "Risk of overfitting since auc scores of validation set and test set differ significantly."


def visualise_tree(decision_tree, feature_names=None, class_names=None):
    """
    Generate GraphicViz representation of decision tree and visualise it in png format (Rf.png).

    Args:
        decision_tree: Decision tree algorithm.
        feature_names (list): list of features names.
        class_names (list): list of class names.

    Returns:
        RF.png: GraphicViz representation of decision tree as a png.

    """
    if class_names is None:
        class_names = ["out", "in"]

    export_graphviz(decision_tree, out_file="RF.dot", feature_names=feature_names, class_names=class_names)
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]

    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot file to produce visualisation.")


def get_feature_importance(fitted_algorithm, initial_parameters):
    """ Show the importance of features in the forest. The higher the score, the higher the importance. """

    importance = fitted_algorithm.best_estimator_.feature_importances_
    indices = np.argsort(importance)[::-1]

    w = window.WindowParameters(initial_parameters=initial_parameters, num_filtering_scales=len(importance))
    mass_window = w.smoothing_masses

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(importance)):
        print("%d. feature %d: importance score %f, top-hat filter mass %e %s" % (f + 1, indices[f], importance[
            indices[f]], mass_window[indices[f]], mass_window.units))

    # Plot the feature importance of the forest
    plot.plot_feature_importance_ranking(importance, indices)


def write_to_file_cv_results(file, cv_estimator):
    f = open(file, "w+")

    f.write("Best parameters set found on test set: \n")
    f.write(str(cv_estimator.best_params_))
    f.write('\n')

    f.write("\nBest estimator:\n")
    f.write(str(cv_estimator.best_estimator_))
    f.write('\n')

    f.write("\nGrid scores on test set:\n")
    means = cv_estimator.cv_results_['mean_test_score']
    stds = cv_estimator.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_estimator.cv_results_['params']):
        f.write("%0.3f (+/-%0.03f) for %r\n"
                % (mean, std * 2, params))
    f.write('\n')

    f.write("\nGrid scores on train set:\n")
    means = cv_estimator.cv_results_['mean_train_score']
    stds = cv_estimator.cv_results_['std_train_score']
    for mean, std, params in zip(means, stds, cv_estimator.cv_results_['params']):
        f.write("%0.3f (+/-%0.03f) for %r\n"
                % (mean, std * 2, params))
    f.close()
