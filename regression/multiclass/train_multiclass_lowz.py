import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# Get training set

# saving_path = "/share/data2/lls/multiclass/lowz/ics_density_only/"
saving_path = "/share/data2/lls/multiclass/lowz/z0_density_only/"

# path_features = "/share/data2/lls/features_w_periodicity_fix/"
# # ics_den_features = np.lib.format.open_memmap(path_features + "ics_density_contrasts.npy", mode="r", shape=(256**3,
# # 50))
# # z8_den_features = np.lib.format.open_memmap(path_features + "z8_density_contrasts.npy", mode="r", shape=(256**3, 50))
# z0_den_features = np.lib.format.open_memmap(path_features + "z0l_density_contrasts.npy", mode="r", shape=(256**3, 50))
#
# training_ind = np.load("/share/data2/lls/multiclass/inertia_plus_den/training_ids.npy")
# testing_ids = np.load("/share/data2/lls/multiclass/inertia_plus_den/testing_ids.npy")
#
# class_labels = np.load("/share/data2/lls/multiclass/inertia_plus_den/class_labels.npy")
# y = label_binarize(class_labels, classes=list(np.arange(class_labels.max() + 1)))
#
# #x_training = np.column_stack((ics_den_features[training_ind], z8_den_features[training_ind]))
# x_training = z0_den_features[training_ind]
# y_training = class_labels[training_ind]
#
# # use cross validation or not
#
# cv = False
#
# if cv is True:
#     param_grid = {"n_estimators": [1000, 1300],
#                   "max_features": ["auto", 0.4, 0.25],
#                   "min_samples_leaf": [15, 5],
#                   }
#
#     class AucScore(object):
#         def __call__(self, estimator, X, y, true_class=1):
#             y_probabilities = estimator.predict_proba(X)
#             return ml.get_auc_score(y_probabilities, y, true_class=true_class)
#
#     rf = RandomForestClassifier(bootstrap=True, min_samples_split=2, n_jobs=60, criterion="gini")
#     auc_scorer = AucScore()
#     random_forest = GridSearchCV(rf, param_grid, scoring=auc_scorer, cv=5, n_jobs=1)
#
# else:
#     # don't use cross validation
#     random_forest = RandomForestClassifier(n_estimators=1300, max_features="auto", bootstrap=True, min_samples_split=5,
#                                            n_jobs=60)
#
# ovr = OneVsRestClassifier(random_forest, n_jobs=1)
# ovr.fit(x_training, y_training)
#
# imp = np.zeros((len(ovr.estimators_), int(x_training.shape[1])))
# for i in range(len(ovr.estimators_)):
#     imp[i] = ovr.estimators_[i].feature_importances_
#
# np.save(saving_path + "f_imp.npy", imp)
#
# # Classify
#
# # X_test = np.column_stack((ics_den_features[testing_ids], z8_den_features[testing_ids]))
# X_test = z0_den_features[testing_ids]
# y_predicted = ovr.predict_proba(X_test)
# np.save(saving_path + "predicted_classes.npy", y_predicted)


# Save fpr and tpr

y_predicted = np.load(saving_path + "predicted_classes.npy")
class_test_set = np.load("/share/data2/lls/multiclass/inertia_plus_den/classes_test_set.npy")
y = label_binarize(class_test_set, classes=list(np.arange(class_test_set.max() + 1)))

FPR = np.zeros((15, 50))
TPR = np.zeros((15, 50))
AUC = np.zeros((15,))
for i in range(15):
    FPR[i], TPR[i], AUC[i], threshold = ml.roc(y_predicted[:,i], y[:,i], true_class=1)

np.save(saving_path + "fpr_tpr_auc_z0_only.npy", np.column_stack((FPR, TPR, AUC)))
