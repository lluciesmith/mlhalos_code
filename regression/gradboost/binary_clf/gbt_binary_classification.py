from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.externals import joblib
import sys
sys.path.append("/home/lls/mlhalos_code")
from mlhalos import machinelearning as ml
from mlhalos import plot

path = "/home/lls/stored_files/GBT_binary_clf/"


def get_multiple_rocs(predicted, true, labels=[""], add_EPS=False, fpr_EPS=None, tpr_EPS=None, add_ellipsoidal=False,
                      fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_EPS=None, label_ellipsoidal=None,
                      fontsize_labels=17):
    FPR = np.zeros((2, 50))
    TPR = np.zeros((2, 50))
    AUC = np.zeros((2, ))
    for i in range(2):
        FPR[i], TPR[i], AUC[i], threshold = ml.roc(predicted[i], true[i], true_class=1)

    fig = plot.roc_plot(FPR.transpose(), TPR.transpose(), AUC, labels=labels, figsize=None, add_EPS=add_EPS,
                        fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS, add_ellipsoidal=add_ellipsoidal,
                        fpr_ellipsoidal=fpr_ellipsoidal, tpr_ellipsoidal=tpr_ellipsoidal, label_EPS=label_EPS,
                        label_ellipsoidal=label_ellipsoidal, frameon=False,
                        fontsize_labels=fontsize_labels, cols=["#8856a7", "#7ea6ce" ])

    return FPR, TPR, AUC, fig

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


den_features = np.load("/share/data1/lls/shear_quantities/density_features.npy")
den_sub_ell = np.load("/share/data1/lls/shear_quantities/den_sub_ellipticity_features.npy")
assert (den_features[:, -1] == den_sub_ell[:, -1]).all()

den_den_sub_ell = np.column_stack((den_features[:, :-1], den_sub_ell))
del den_features
del den_sub_ell

training_features = get_training_features(den_den_sub_ell)
testing_features = get_testing_features(den_den_sub_ell)
del den_den_sub_ell

# den + shear

# clf = GradientBoostingClassifier(n_estimators=1000, max_features=0.5, max_depth=30, subsample=0.6)
# clf.fit(training_features[:, :-1], training_features[:, -1])
# joblib.dump(clf, path + "den_plus_ell/clf.pkl", compress=3)
#
# y_predicted = clf.predict_proba(testing_features[:, :-1])
# y_true = testing_features[:, -1]
#
# np.save(path + "den_plus_ell/predicted_den+den_sub_ell.npy", y_predicted)
# np.save(path + "den_plus_ell/true_den+den_sub_ell.npy", y_true)

# den only

clf = GradientBoostingClassifier(n_estimators=500, max_features=0.5, max_depth=30, subsample=0.6)
# clf = GradientBoostingClassifier(n_estimators=1300, min_samples_leaf=15, min_samples_split=2,
#                              n_jobs=3, criterion='entropy', max_features=0.4)
clf.fit(training_features[:, :50], training_features[:, -1])

joblib.dump(clf, path + "den_only/clf.pkl", compress=3)

y_predicted = clf.predict_proba(testing_features[:, :50])
y_true = testing_features[:, -1]

np.save(path + "den_only/predicted_den.npy", y_predicted)
np.save(path + "den_only/true_den.npy", y_true)




########### PLOT ROCS AND COMPARE TO RF ###########

pred_shear_RF = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol/"
                        "predicted_den+den_sub_ell+den_sub_prol.npy")
true_shear_RF = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/den+den_sub_ell+den_sub_prol/"
                        "true_den+den_sub_ell+den_sub_prol.npy")

pred_shear_GBT = np.load("/Users/lls/Documents/mlhalos_files/stored_files/GBT/den_plus_shear/predicted_den+den_sub_ell.npy")
true_shear_GBT = np.load("/Users/lls/Documents/mlhalos_files/stored_files/GBT/den_plus_shear/true_den+den_sub_ell.npy")

pred_density_RF = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/"
                       "predicted_den.npy")
true_density_RF = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/true_den.npy")

pred_den_GBT = np.load("/Users/lls/Documents/mlhalos_files/stored_files/GBT/den_only/predicted_den.npy")
true_den_GBT = np.load("/Users/lls/Documents/mlhalos_files/stored_files/GBT/den_only/true_den.npy")

pred_all = np.array([pred_shear_GBT, pred_shear_RF])
true_all = np.array([true_shear_GBT, true_shear_RF])
np.testing.assert_allclose(true_shear_GBT, true_shear_RF)

fig_RF_GBT = get_multiple_rocs(pred_all, true_all, labels=["GBT", "RF"])

pred_all = np.array([pred_den_GBT, pred_density_RF])
true_all = np.array([true_den_GBT, true_density_RF])
get_multiple_rocs(pred_all, true_all, labels=["GBT", "RF"])