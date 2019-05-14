from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib

path = "/share/data1/lls/shear_quantities/"

def get_testing_index():
    training_index = np.load("/home/lls/stored_files/50k_features_index.npy")
    testing_index = ~np.in1d(np.arange(256 ** 3), training_index)
    return testing_index

# def get_training_features(features):
#     training_index = np.load("/home/lls/stored_files/50k_features_index.npy")
#     training_features = features[training_index]
#     return training_features
#
# def get_testing_features(features):
#     testing_index = get_testing_index()
#     testing_features = features[testing_index]
#     return testing_features

training_index = np.load("/home/lls/stored_files/50k_features_index.npy")
shape_0 = 256**3

# den_features = np.load(path + "density_features.npy")
den_features = np.lib.format.open_memmap(path + "density_features.npy", mode="r", shape=(shape_0, 51))
den_training = den_features[training_index]
del den_features

den_sub_ell = np.lib.format.open_memmap(path + "den_sub_ellipticity_features.npy", mode="r", shape=(shape_0, 51))
ell_training = den_sub_ell[training_index]
del den_sub_ell

den_sub_prol = np.lib.format.open_memmap(path + "den_sub_prolateness_features.npy", mode="r", shape=(shape_0, 51))
prol_training = den_sub_prol[training_index]
del den_sub_prol

# assert (den_features[:, -1] == den_sub_ell[:, -1]).all()
# assert (den_features[:, -1] == den_sub_prol[:, -1]).all()

training_features = np.column_stack((den_training[:,:-1], ell_training[:, :-1], prol_training))
print(training_features.shape)

clf = RandomForestClassifier(n_estimators=1300, min_samples_leaf=15, min_samples_split=2,
                             n_jobs=60, criterion='entropy', max_features="auto")
clf.fit(training_features[:, :-1], training_features[:, -1])
del training_features

print(clf.feature_importances_)
np.save(path + "classifier_den+den_sub_ell+den_sub_prol/feature_importances_den+den_sub_ell+den_sub_prol_UPGRADE.npy",
        clf.feature_importances_)

joblib.dump(clf, path + "classifier_den+den_sub_ell+den_sub_prol/clf_upgraded.pkl", compress=3)

# y_predicted = clf.predict_proba(testing_features[:, :-1])
# y_true = testing_features[:, -1]

# np.save(path + "predicted_den+den_sub_ell+den_sub_prol.npy", y_predicted)
# np.save(path + "true_den+den_sub_ell+den_sub_prol.npy", y_true)
