import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from sklearn.externals import joblib


def find_predicted_and_true_labels(trained_classifier, features):
    predicted_probabilities = trained_classifier.predict_proba(features[:, :-1])
    true_labels = features[:, -1]
    return predicted_probabilities, true_labels


try:
    test_features = np.load("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/test_density_shear_features.npy")

except IOError:
    features_all = np.load("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/density_full_shear_features.npy")
    index_training = np.load("/home/lls/stored_files/50k_features_index.npy")

    test_features = features_all[~np.in1d(np.arange(len(features_all)), index_training)]
    np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/test_density_shear_features.npy", test_features)


classifier = joblib.load("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/classifier/classifier.pkl")
predicted_proba, true_labels = find_predicted_and_true_labels(classifier, test_features)

np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/predicted_probabilities.npy", predicted_proba)
np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/true_labels.npy", true_labels)


