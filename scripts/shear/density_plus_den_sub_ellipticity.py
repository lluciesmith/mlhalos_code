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
den_sub_ell = np.load(path + "den_sub_ellipticity_features.npy")
assert (den_features[:, -1] == den_sub_ell[:, -1]).all()

den_den_sub_ell = np.column_stack((den_features[:,:-1], den_sub_ell))
del den_features
del den_sub_ell

training_features = get_training_features(den_den_sub_ell)
testing_features = get_testing_features(den_den_sub_ell)
del den_den_sub_ell

clf = RandomForestClassifier(n_estimators=1300, min_samples_leaf=15, min_samples_split=2,
                             n_jobs=3, criterion='entropy', max_features=0.4)
clf.fit(training_features[:, :-1], training_features[:, -1])
del training_features
joblib.dump(clf, path + "classifier_den+den_sub_ell/clf.pkl", compress=3)

y_predicted = clf.predict_proba(testing_features[:, :-1])
y_true = testing_features[:, -1]

np.save(path + "predicted_den+den_sub_ell.npy", y_predicted)
np.save(path + "true_den+den_sub_ell.npy", y_true)