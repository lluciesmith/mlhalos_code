
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code")
import numpy as np
from regression.feature_importances import functions_imp_tests as it
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statistics import mode
from sklearn.model_selection import GridSearchCV
from statsmodels import robust


# First load the training set and testing set to use for the feature importances tests

training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/training_ids.npy")
log_mass_training = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                            "files/log_halo_mass_training.npy")

testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/files/testing_ids.npy")
log_halo_mass_testing = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/"
                                "files/log_halo_mass_testing.npy")


# training features
train_feat_04_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                             "/files/training_feat_04_correlation.npy")
train_feat_07_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                             "/files/training_feat_07_correlation.npy")

test_feat_04_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                            "/files/testing_feat_04_correlation.npy")
test_feat_07_corr = np.load("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests"
                            "/files/testing_feat_07_correlation.npy")

training_features_all = np.column_stack((train_feat_07_corr, train_feat_04_corr, log_mass_training))
testing_features_all = np.column_stack((test_feat_07_corr, test_feat_04_corr))

# predictions

rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=10)

random_grid = {'max_depth': [100, 200, None],  'max_features': [5, 10, 30]}

rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, cv=3, verbose=2, n_jobs=-1,
                         scoring="neg_mean_squared_error")
rf_random.fit(training_features_all[:, :-1], training_features_all[:, -1])

CV_test_score = rf_random.cv_results_['mean_test_score']
CV_train_score = rf_random.cv_results_['mean_train_score']

print("CV test scores are " + str(rf_random.cv_results_['mean_test_score'])
      + " and CV train scores are " + str(rf_random.cv_results_['mean_train_score']))

diff_train_test = ((CV_train_score - CV_test_score)/(CV_train_score + CV_test_score))[rf_random.cv_results_[
                                                                                          'rank_test_score'] == 1] * 100

print("For the best parameter, the difference between train and test score is of " + str(diff_train_test) + " %.")

# explore distributions at leaf nodes that lead to test set predictions

small_mass = np.where((log_halo_mass_testing <= 11) & (log_halo_mass_testing >=10))[0]
particle_ids = np.random.choice(small_mass, 100, replace=False)

leaf_nodes_test_set = rf_random.best_estimator_.apply(testing_features_all)
leaf_nodes_train_set = rf_random.best_estimator_.apply(training_features_all[:, :-1])

def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

for particle_id in particle_ids:

    mode_each_tree = []
    mean_each_tree = []
    median_each_tree = []
    weights = []
    mad_array =[]
    for i in range(rf_random.best_estimator_.n_estimators):
        training_ids_in_leaf_node_tree_i = np.where(leaf_nodes_train_set[:, i] == leaf_nodes_test_set[particle_id, i])[0]
        # print("There are " + str(len(training_ids_in_leaf_node_tree_i)) + " training particles in leaf node of tree "
        #       + str(i))
        try:
            m, n = mode(log_mass_training[training_ids_in_leaf_node_tree_i])
        except:
            m = np.median(log_mass_training[training_ids_in_leaf_node_tree_i])

        m_i = mad(log_mass_training[training_ids_in_leaf_node_tree_i])
        mad_array.append(m_i)
        weights.append(len(training_ids_in_leaf_node_tree_i))
        mode_each_tree.append(m)
        mean_each_tree.append(np.mean(log_mass_training[training_ids_in_leaf_node_tree_i]))
        median_each_tree.append(np.median(log_mass_training[training_ids_in_leaf_node_tree_i]))

    weighted_median = np.repeat(median_each_tree, weights)

    weights = 1 - np.array(mad_array)
    weighted_mean = np.sum(weights * median_each_tree) / np.sum(weights)

    plt.figure()
    plt.hist(median_each_tree, bins=20, histtype="step", color="b", normed=True)
    # plt.hist(weighted_median, bins=20, histtype="step", color="r", ls="--", normed=True)
    #plt.hist(mode_each_tree, bins=20, histtype="step", label="mode", color="g")
    plt.hist(mean_each_tree, bins=20, histtype="step", label="mean", color="r", normed=True)
    plt.axvline(log_halo_mass_testing[particle_id], lw=2, color="k")
    plt.xlabel("Predicted mass")

    plt.axvline(np.median(median_each_tree), color="b")
    # try:
    #     plt.axvline(mode(median_each_tree), color="g", label="mode")
    # except:
    #     pass
    plt.axvline(weighted_mean, color="r", label="weighted")
    plt.legend(loc="best")
    plt.show()
    plt.savefig("/Users/lls/Documents/mlhalos_files/regression/feature_importances_tests/estimators_particle_" + str(
        particle_id) + ".png")
    plt.clf()

    # weight medians by number of leaves in the tree

####### COMPARE MEAN OF MEANS VS MEDIANS OF MEDIANS FOR SMALL MASS HALOS ######

small_mass = np.where((log_halo_mass_testing <= 11))[0]
leaf_nodes_train_set = rf_random.apply(train_feat_04_corr)
leaf_nodes_test_set = rf_random.apply(test_feat_04_corr)

mean_mean = []
median_median = []
mean_median = []
median_mean = []

for particle_id in small_mass[12584:]:

    mean_each_tree = []
    median_each_tree = []

    for i in range(rf_random.n_estimators):
        training_ids_in_leaf_node_tree_i = np.where(leaf_nodes_train_set[:, i] == leaf_nodes_test_set[particle_id, i])[0]
        median_i = np.median(log_mass_training[training_ids_in_leaf_node_tree_i])

        mean_each_tree.append(np.mean(log_mass_training[training_ids_in_leaf_node_tree_i]))
        median_each_tree.append(median_i)

    mean_mean.append(np.mean(mean_each_tree))
    median_mean.append(np.median(mean_each_tree))
    mean_median.append(np.mean(median_each_tree))
    median_median.append(np.median(median_each_tree))


# Check that mean of means is the same as random forest prediction!

plt.scatter(log_halo_mass_testing[small_mass], mean_mean, color="k", label="mean(mean)")
plt.scatter(log_halo_mass_testing[small_mass], median_median, color="b", label="median(median)")
plt.scatter(log_halo_mass_testing[small_mass], mean_median, color="g", label="mean(median)")
plt.scatter(log_halo_mass_testing[small_mass], median_mean, color="r", label="median(mean)")

plt.figure()
plt.hist(median_median, bins=20, histtype="step", color="b",  label="median(median)")
plt.hist(mean_mean, bins=20, histtype="step", color="g", label="mean(mean)")
plt.hist(log_halo_mass_testing[small_mass], bins=10, histtype="step", color="k", label="true")
plt.legend(loc="best")


small_mass = np.where((log_halo_mass_testing <= 11) & (log_halo_mass_testing >=10))[0]
particle_ids = np.random.choice(small_mass, 100, replace=False)

leaf_nodes_test_set = rf_random.best_estimator_.apply(testing_features_all)
leaf_nodes_train_set = rf_random.best_estimator_.apply(training_features_all[:, :-1])


n = np.where((log_halo_mass_testing[small_mass]>= 10.75))[0]
# n = np.where((log_halo_mass_testing[small_mass] <= 10.5))[0]
plt.figure()
plt.hist(np.array(median_median)[n], bins=20, histtype="step", color="b", label="median(median)")
plt.hist(np.array(mean_mean)[n], bins=20, histtype="step", color="g", label="mean(mean)")
plt.hist(log_halo_mass_testing[small_mass][n], bins=10, histtype="step", color="k", label="true")
plt.legend(loc="best")

####### COMPARE MEAN OF MEANS VS MEDIANS OF MEDIANS FOR SMALL MASS HALOS ######



