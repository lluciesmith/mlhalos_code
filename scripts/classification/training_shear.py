import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from mlhalos import machinelearning as ml

# path = sys.argv[2]
# number_of_cores = sys.argv[3]
#
# if path == "hypatia":
#     path = "/home/lls/stored_files"
# elif path == "macpro":
#     sys.path.append("/Users/lls/Documents/mlhalos_code/scripts")
#     path = "/Users/lls/Documents/CODE"

try:
    training_den_shear_features = np.load("/home/lls/stored_files/shear_and_density/training_density_shear_features.npy")

except IOError:
    density_shear_features = np.load("/home/lls/stored_files/shear_and_density/density_shear_features.npy")
    index_training = np.load("/home/lls/stored_files/50k_features_index.npy")

    training_den_shear_features = density_shear_features[index_training, :]
    np.save("/home/lls/stored_files/shear_and_density/training_density_shear_features.npy", training_den_shear_features)


trained_algo = ml.MLAlgorithm(training_den_shear_features, split_data_method=None, n_jobs=60,
                              save=True, path="/home/lls/stored_files/shear_and_density/classifier/classifier.pkl")

print(trained_algo.classifier.best_score_)
print(trained_algo.classifier.best_estimator_)
print(trained_algo.classifier.best_params_)

np.save("/home/lls/stored_files/shear_and_density/feature_importances.npy", trained_algo.feature_importances)

