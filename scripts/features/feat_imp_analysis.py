"""
In this script, I would like to test the reliability of feature importances plots.

I compute three tests:

1. Add a random number as an extra feature (feature 50 in the plot) to the set of 50 features of a 50,000 random
sample of particles from the full box --> Check that this feature is an irrelevant feature in the feature importance
plot.
2. Add the ground truth of the training set particles as an extra feature (feature 50 in the plot) --> Check that
this feature is the most relevant in the feature importance plot.
3. Add the most relevant feature of the original 50 features set as an extra feature --> We expect to see the two
identical features having equal importance and it being equal to half its importance in the original set.

"""


import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt

from mlhalos import distinct_colours
from mlhalos import machinelearning as ml


features_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features.npy")

# Add a random feature

rand = []
for i in range(50000):
    rand.append(np.random.randint(1, 10000))

rand = np.array(rand)
rand = rand.reshape(-1, 1)

rand = StandardScaler().fit_transform(rand)

features_1 = np.column_stack((features_training[:, :-1], rand))
features_w_rand_feat = np.column_stack((features_1, features_training[:, -1]))

algorithm_w_rand_feat = ml.MLAlgorithm(features_w_rand_feat)

importance_w_rand_feat = algorithm_w_rand_feat.classifier.best_estimator_.feature_importances_
indices_w_rand_feat = np.argsort(importance_w_rand_feat)[::-1]

np.save('/Users/lls/Desktop/feature_importances/importance_random_feature.npy',
        importance_w_rand_feat)
np.save('/Users/lls/Desktop/feature_importances/indices_random_feature.npy',
        indices_w_rand_feat)


# Add ground truth

features_training_truth = features_training[:, -1]

features_2 = np.column_stack((features_training[:, :-1], features_training_truth))
features_w_truth = np.column_stack((features_2, features_training[:, -1]))

algorithm_w_truth = ml.MLAlgorithm(features_w_truth)

importance_w_truth = algorithm_w_truth.classifier.best_estimator_.feature_importances_
indices_w_truth = np.argsort(importance_w_truth)[::-1]
np.save('/Users/lls/Desktop/feature_importances/importance_truth_feature.npy',
        importance_w_truth)
np.save('/Users/lls/Desktop/feature_importances/indices_truth_feature.npy',
        indices_w_truth)


# Add feature 27 ( most relevant) again

feature_27 = features_training[:,27]
features_3 = np.column_stack((features_training[:, :-1], feature_27))

features_plus_27 = np.column_stack((features_3, features_training[:, -1]))

algorithm_w_27 = ml.MLAlgorithm(features_plus_27)

importance_w_27 = algorithm_w_27.classifier.best_estimator_.feature_importances_
indices_w_27 = np.argsort(importance_w_27)[::-1]

np.save('/Users/lls/Desktop/feature_importances/importance_feature_w_27.npy',
        importance_w_27)
np.save('/Users/lls/Desktop/feature_importances/indices_feature_w_27.npy',
        indices_w_27)

############# PLOTS ################

# PLOT SCORES vs TOP-LEVEL SPLIT FOR EACH TREE IN THE TRAINING SET CASE OF ORIGINAL TRAINING SET

top_level_trees_27 = np.load("/Users/lls/Desktop/plot_scores_level/top_level_trees_27.npy")
top_level_trees_20 = np.load("/Users/lls/Desktop/plot_scores_level/top_level_trees_20.npy")
top_level_trees_3 = np.load("/Users/lls/Desktop/plot_scores_level/top_level_trees_3.npy")

scores_27 = np.load("/Users/lls/Desktop/plot_scores_level/scores_27.npy")
scores_20 = np.load("/Users/lls/Desktop/plot_scores_level/scores_20.npy")
scores_3 = np.load("/Users/lls/Desktop/plot_scores_level/scores_3.npy")

unique_level_27 = np.array(sorted(set(top_level_trees_27)))
unique_level_20 = np.array(sorted(set(top_level_trees_20)))
unique_level_3 = np.array(sorted(set(top_level_trees_3)))

mean_27 = []
std_27 = []
for i in unique_level_27:
    if len(scores_27[top_level_trees_27 == i]) > 3:
        mean = np.mean(scores_27[top_level_trees_27 == i])
        std = np.std(scores_27[top_level_trees_27 == i])
        mean_27.append(mean)
        std_27.append(std)
    else:
        mean_27.append(np.mean(scores_27[top_level_trees_27 == i]))
        std_27.append(0)

mean_20 = []
std_20 = []
for i in unique_level_20:
    if len(scores_20[top_level_trees_20 == i]) > 3:
        mean = np.mean(scores_20[top_level_trees_20 == i])
        std = np.std(scores_20[top_level_trees_20==i])
        mean_20.append(mean)
        std_20.append(std)
    else:
        mean_20.append(np.mean(scores_20[top_level_trees_20 == i]))
        std_20.append(0)

mean_3 = []
std_3 = []
for i in unique_level_3:
    if len(scores_3[top_level_trees_3 == i]) > 3:
        mean = np.mean(scores_3[top_level_trees_3 == i])
        std = np.std(scores_3[top_level_trees_3 == i])
        mean_3.append(mean)
        std_3.append(std)
    else:
        mean_3.append(np.mean(scores_3[top_level_trees_3 == i]))
        std_3.append(0)

std_27 = np.array(std_27)
std_20 = np.array(std_20)
std_3 = np.array(std_3)
mean_3 = np.array(mean_3)
mean_27 = np.array(mean_27)
mean_20 = np.array(mean_20)


def plot():
    cols = distinct_colours.get_distinct(3)
    plt.figure(figsize=(17, 7))
    plt.errorbar(unique_level_27[std_27!=0], mean_27[std_27!=0], yerr=std_27[std_27!=0], label="high relevance", color=cols[0],
                 fmt='o', ms=5)
    plt.scatter(unique_level_27[std_27==0], mean_27[std_27==0], marker='o', color=cols[0], s=100)

    plt.errorbar(np.array(unique_level_20)[std_20!=0]-0.1, mean_20[std_20!=0], yerr=std_20[std_20!=0], color=cols[1], label="mid " \
                                                                                                                  "relevance",
                 fmt='o', ms=5)
    plt.scatter(np.array(unique_level_20)[std_20==0]-0.1, mean_20[std_20==0], color=cols[1],
                marker='o', s=100)

    plt.errorbar(np.array(unique_level_3)[std_3!=0]+0.1, mean_3[std_3!=0], yerr=std_3[std_3!=0], label="low relevance", color=cols[
        2], fmt='o',
                 ms=5)
    plt.scatter(np.array(unique_level_3)[std_3==0]+0.1, mean_3[std_3==0],  color=cols[2],
                marker='o', s=100)

    plt.legend(fontsize=17.0)
    plt.yscale("log")
    plt.ylabel("Importance score")
    plt.xlabel("Tree level")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20.0)
    plt.savefig("/Users/lls/Desktop/mean.png")

plot()
