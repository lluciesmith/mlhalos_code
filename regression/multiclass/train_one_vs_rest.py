import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# Get training set

saving_path = "/share/data2/lls/multiclass/inertia_plus_den/one_vs_rest/CV/"


path_traj = "/share/data1/lls/shear_quantities/quantities_id_ordered/"
den_features = np.lib.format.open_memmap(path_traj + "density_trajectories.npy", mode="r", shape=(256**3, 50))

path_inertia = "/share/data1/lls/regression/inertia/cores_40/"
eig_0 = np.lib.format.open_memmap(path_inertia + "eigenvalues_0.npy", mode="r", shape=(256**3, 50))

training_ind = np.load("/share/data2/lls/multiclass/inertia_plus_den/training_ids.npy")
testing_ids = np.load("/share/data2/lls/multiclass/inertia_plus_den/testing_ids.npy")

class_labels = np.load("/share/data2/lls/multiclass/inertia_plus_den/class_labels.npy")
y = label_binarize(class_labels, classes=list(np.arange(class_labels.max() + 1)))

x_training = np.column_stack((den_features[training_ind], eig_0[training_ind]))
y_training = class_labels[training_ind]

# use cross validation

cv = True
param_grid = {"n_estimators": [1000, 1300],
              "max_features": ["auto", 0.4, 0.25],
              "min_samples_leaf": [15, 5],
              }


class AucScore(object):
    def __call__(self, estimator, X, y, true_class=1):
        y_probabilities = estimator.predict_proba(X)
        return ml.get_auc_score(y_probabilities, y, true_class=true_class)

rf = RandomForestClassifier(bootstrap=True, min_samples_split=2, n_jobs=60, criterion="gini")
auc_scorer = AucScore()
random_forest = GridSearchCV(rf, param_grid, scoring=auc_scorer, cv=5, n_jobs=1)

# don't use cross validation
# random_forest = RandomForestClassifier(n_estimators=1000, max_features="auto", bootstrap=True, min_samples_split=2,
#                                        n_jobs=60)

ovr = OneVsRestClassifier(random_forest, n_jobs=1)
ovr.fit(x_training, y_training)

imp = np.zeros((len(ovr.estimators_), 100))
for i in range(len(ovr.estimators_)):
    imp[i] = ovr.estimators_[i].feature_importances_

np.save(saving_path + "f_imp.npy", imp)

# classify
X_test = np.column_stack((den_features[testing_ids], eig_0[testing_ids]))
y_predicted = ovr.predict_proba(X_test)
np.save(saving_path + "predicted_classes.npy", y_predicted)