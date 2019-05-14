import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from mlhalos import machinelearning as ml


try:
    training_den_shear_features = np.load(
        "/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/training_density_full_shear_features"
        ".npy")

except IOError:
    density_shear_features = np.load(
        "/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/density_full_shear_features.npy")
    index_training = np.load("/home/lls/stored_files/50k_features_index.npy")

    training_den_shear_features = density_shear_features[index_training, :]
    np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/training_density_full_shear_features.npy",
            training_den_shear_features)

# Try ignore outliers in feature distributions.
# If ellipticity and prolateness are greater than 10 --> give them the values of the median

# for i in range(50, 100):
#     pos = np.where(abs(training_den_shear_features[:, i]) > 10)[0]
#     training_den_shear_features[pos, i] = np.median(training_den_shear_features[:,i])

# for i in range(50):
#     plt.hist(training_den_shear_features[np.where(training_den_shear_features[:,-1]==1)[0], i], label="in",
#              normed=True, histtype="step", bins=30)
#     plt.hist(training_den_shear_features[np.where(training_den_shear_features[:, -1] == -1)[0], i], label="out",
#              normed=True, histtype="step", bins=30)
#     plt.xlabel("feature " + str(i))
#     plt.legend(loc="best")
#     plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/distributions_50k/feature_" + str(i) +".pdf")
#     plt.clf()

# training_den_shear_features = np.column_stack((training_den_shear_features[:,50:100], training_den_shear_features[:,
#                                                                                       -1]))

trained_algo = ml.MLAlgorithm(training_den_shear_features, cross_validation=True, split_data_method=None, n_jobs=60,
                              save=True,
                              path="/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/"
                                   "classifier/classifier.pkl"
                              )

print(trained_algo.classifier.best_score_)
print(trained_algo.classifier.best_estimator_)
print(trained_algo.classifier.best_params_)

np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/feature_importances.npy",
        trained_algo.feature_importances)
