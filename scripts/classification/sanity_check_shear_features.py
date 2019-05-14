import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlhalos import machinelearning as ml
from sklearn.externals import joblib

training_den_shear_features = np.load("/home/lls/stored_files/shear_and_density/training_density_shear_features.npy")
test_features = np.load("/home/lls/stored_files/shear_and_density/test_density_shear_features.npy")

ind = np.random.choice(range(len(test_features)), 50000)

print("loaded stuff")

training = training_den_shear_features[:, :50]
testing = test_features[ind, :50]
train_labels = training_den_shear_features[:, -1]
test_labels = test_features[ind, -1]

print("extract density stuff")

del training_den_shear_features
del test_features

classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                    max_depth=None, max_features="auto", max_leaf_nodes=None, min_samples_leaf=15,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=1300, n_jobs=60,
                                    oob_score=False, random_state=None, verbose=0, warm_start=False)

classifier = classifier.fit(training, train_labels)

print(classifier.feature_importances_)
np.save("/home/lls/stored_files/shear_and_density/density_plus_ell0_importances.npy", classifier.feature_importances_)

print("fitted")

#joblib.dump(classifier, "/home/lls/stored_files/shear_and_density/sanity_check/classifier.pkl")


#del train_prolatness
#del train_labels

#classifier = joblib.load("/home/lls/stored_files/shear_and_density/sanity_check/classifier.pkl")

predicted = classifier.predict_proba(testing)

print("predicted")

#np.save("/home/lls/stored_files/shear_and_density/sanity_check/predicted.npy", predicted)
true = test_labels

#np.save("/home/lls/stored_files/shear_and_density/sanity_check/true.npy", true)

auc = ml.get_auc_score(predicted, true)
print("The auc score is " + str(auc))

#np.save("/home/lls/stored_files/shear_and_density/auc_just_prolateness.npy", np.array([auc]))


