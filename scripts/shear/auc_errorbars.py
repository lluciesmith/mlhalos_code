import sys
sys.path.append("/home/lls/mlhalos_code")
#sys.path.append("/Users/lls/Documents/mlhalos_code")

import numpy as np
from mlhalos import machinelearning as ml
from sklearn.ensemble import RandomForestClassifier

path = "/home/lls/stored_files/shear_quantities/"
# path = "/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/"


def train_and_test(training_features, testing_features):
    clf = RandomForestClassifier(n_estimators=1300, min_samples_leaf=15, min_samples_split=2, n_jobs=60,
                                 criterion='entropy', max_features="auto")

    clf.fit(training_features[:, :-1], training_features[:, -1])

    y_pred = clf.predict_proba(testing_features[:, :-1])
    y_true = testing_features[:, -1]
    return y_pred, y_true


def get_ten_aucs_for_same_training_set(features):
    AUC = np.zeros((10,))

    a = np.arange(256 ** 3)

    for i in range(10):
        testing_index = np.random.choice(range(256 ** 3), 100000)
        all_no_testing = a[~np.in1d(a, testing_index)]
        training_index = np.random.choice(all_no_testing, 50000)

        training_features = features[training_index, :]
        testing_features = features[testing_index, :]

        y_predicted, y_true = train_and_test(training_features, testing_features)

        del training_features
        del training_index

        fpr, tpr, auc, threshold = ml.roc(y_predicted, y_true)
        AUC[i] = auc

        del y_predicted
        del y_true
        del fpr, tpr, auc, threshold

    return AUC


testing_index = np.random.choice(range(256**3), 40000)

######### AUC of 10 runs for density ##########

den_features = np.load(path + "density_features.npy")

aucs_den = get_ten_aucs_for_same_training_set(den_features)
np.save(path + "aucs_density_only.npy", aucs_den)

del aucs_den

######### AUC of 10 runs for density + density-subtracted ellipticity + den_sub_prolateness ##########

den_sub_ell = np.load(path + "den_sub_ellipticity_features.npy")
den_sub_prol = np.load(path + "den_sub_prolateness_features.npy")

features = np.column_stack((den_features[:, :-1], den_sub_ell[:, :-1], den_sub_prol[:, :]))

del den_features
del den_sub_ell
del den_sub_prol

aucs_den_den_sub_ell_prol = get_ten_aucs_for_same_training_set(features)
np.save(path + "aucs_density_plus_den_sub_ell_prol.npy", aucs_den_den_sub_ell_prol)

del features
del aucs_den_den_sub_ell_prol


######### AUC of 10 runs for eigenvalues ##########

eigenvalues = np.load(path + "eigenvalues_features.npy")


aucs_eigenvalues = get_ten_aucs_for_same_training_set(eigenvalues)
np.save(path + "aucs_eigenvalues.npy", aucs_eigenvalues)
