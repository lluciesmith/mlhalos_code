import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib

path = "/home/lls/stored_files/shear_quantities/"


def get_testing_index():
    training_index = np.load("/home/lls/stored_files/50k_features_index.npy")
    testing_index = ~np.in1d(np.arange(256 ** 3), training_index)
    return testing_index

def get_training_features(features):
    training_index = np.load("/home/lls/stored_files/50k_features_index.npy")
    training_features = features[training_index]
    return training_features

def get_testing_features(features):
    testing_index = get_testing_index()
    testing_features = features[testing_index]
    return testing_features


den_features = np.load(path + "density_features.npy")

training_features = get_training_features(den_features)
testing_features = get_testing_features(den_features)
del den_features

clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=15, min_samples_split=2,
                             n_jobs=3, criterion='entropy', max_features=0.4)
clf.fit(training_features[:, :-1], training_features[:, -1])

np.save("/share/data1/lls/shear_quantities/importances_density_only", clf.feature_importances_)
