import sys
sys.path.append("/home/lls/mlhalos_code")
#sys.path.append("/Users/lls/Documents/mlhalos_code")

import numpy as np
from mlhalos import machinelearning as ml
from sklearn.ensemble import RandomForestClassifier

path = "/home/lls/stored_files/shear_quantities/"

den_sub_ell = np.load(path + "den_sub_ellipticity_features.npy")
den_sub_prol = np.load(path + "den_sub_prolateness_features.npy")

features = np.column_stack((den_sub_ell[:,:-1], den_sub_prol))

training_index = np.load("/home/lls/stored_files/50k_features_index.npy")
all_index = np.arange(256**3)
testing_index = all_index[~np.in1d(all_index, training_index)]

X = features[training_index,:-1]
y = features[training_index, -1]
X_test = features[testing_index, :-1]
y_test = features[testing_index, -1]

clf = RandomForestClassifier(n_estimators=1300, min_samples_leaf=15, min_samples_split=2,
                             n_jobs=60, criterion='entropy', max_features="auto")
clf.fit(X, y)

y_predicted = clf.predict_proba(X_test)
np.save(path + "smp/predicted_den_sub_ell_prol.npy", y_predicted)
np.save(path + "smp/true_den_sub_ell_prol.npy", y_test)

fpr, tpr, auc, threshold = ml.roc(y_predicted, y_test)
print(auc)

